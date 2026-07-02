import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
from functools import partial
import contextlib
from absl import app, flags

flags.DEFINE_integer('inference_timesteps', 128, 'Number of timesteps for inference.')
flags.DEFINE_integer('inference_generations', 4096, 'Number of generations for inference.')
flags.DEFINE_float('inference_cfg_scale', 1.0, 'CFG scale for inference.')
flags.DEFINE_integer('inference_class_label', -1, 'Specific class label for conditional generation (-1 for random).')
flags.DEFINE_string('samples_dir', None, 'Directory to save generated raw png samples.')

def do_inference(
    FLAGS,
    train_state,
    step,
    dataset,
    dataset_valid,
    shard_data,
    vae_encode,
    vae_decode,
    update,
    get_fid_activations,
    imagenet_labels,
    visualize_labels,
    fid_from_stats,
    truth_fid_stats,
):
    with contextlib.nullcontext():
        global_device_count = jax.device_count()
        key = jax.random.PRNGKey(42 + jax.process_index())
        batch_images, batch_labels = next(dataset)
        valid_images, valid_labels = next(dataset_valid)
        if FLAGS.model.use_stable_vae:
            batch_images = vae_encode(key, batch_images)
            valid_images = vae_encode(key, valid_images)
        batch_labels_sharded, valid_labels_sharded = shard_data(batch_labels, valid_labels)
        labels_uncond = shard_data(jnp.ones(batch_labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes']) # Null token
        eps = jax.random.normal(key, batch_images.shape)

        def process_img(img):
            if FLAGS.model.use_stable_vae:
                img = vae_decode(img[None])[0]
            img = img * 0.5 + 0.5
            img = jnp.clip(img, 0, 1)
            img = np.array(img)
            return img
        
        @partial(jax.jit, static_argnums=(5,))
        def call_model(train_state, images, t, dt, labels, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model
            output = call_fn(images, t, dt, labels, train=False)
            return output
        
        if FLAGS.mode == 'interpolate':
            seed = 5
            eps0 = jax.random.normal(jax.random.PRNGKey(seed), batch_images[0].shape)
            eps1 = jax.random.normal(jax.random.PRNGKey(seed+1), batch_images[0].shape)
            labels = jnp.ones(FLAGS.batch_size,).astype(jnp.int32) * 555
            i = jnp.linspace(0, 1, FLAGS.batch_size)
            i_neg = np.sqrt(1-i**2)
            x = eps0[None] * i_neg[:, None, None, None] + eps1[None] * i[:, None, None, None]
            t_vector = jnp.full((FLAGS.batch_size, ), 0)
            dt_vector = jnp.zeros_like(t_vector)
            cfg_scale = FLAGS.inference_cfg_scale
            v = call_model(train_state, x, t_vector, dt_vector, labels)
            x = x + v * 1.0
            x = vae_decode(x) # Image is in [-1, 1] space.
            x_render = np.array(jax.experimental.multihost_utils.process_allgather(x))
            os.makedirs(FLAGS.save_dir, exist_ok=True)
            np.save(FLAGS.save_dir + f'/x_render.npy', x_render)
            breakpoint()

        denoise_timesteps = FLAGS.inference_timesteps
        num_generations = FLAGS.inference_generations
        cfg_scale = FLAGS.inference_cfg_scale
        x0 = []
        x1 = []
        lab = []
        x_render = []
        activations = []
        images_shape = batch_images.shape
        print(f"Calc FID for CFG {cfg_scale} and denoise_timesteps {denoise_timesteps}")
        for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
            key = jax.random.PRNGKey(42)
            key = jax.random.fold_in(key, fid_it)
            key = jax.random.fold_in(key, jax.process_index())
            eps_key, label_key = jax.random.split(key)
            x = jax.random.normal(eps_key, images_shape)
            if FLAGS.inference_class_label >= 0:
                labels = jnp.full((images_shape[0],), FLAGS.inference_class_label, dtype=jnp.int32)
            else:
                labels = jax.random.randint(label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
            x, labels = shard_data(x, labels)
            x0.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
            delta_t = 1.0 / denoise_timesteps
            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((images_shape[0], ), t)
                if FLAGS.model.train_type == 'naive':
                    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
                    dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow # Smallest dt.
                else: # shortcut
                    dt_flow = np.log2(FLAGS.model['denoise_timesteps'] // denoise_timesteps).astype(jnp.int32)
                    dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow
                    # print(dt_base)
                t_vector, dt_base = shard_data(t_vector, dt_base)
                if cfg_scale == 1:
                    v = call_model(train_state, x, t_vector, dt_base, labels)
                elif cfg_scale == 0:
                    v = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                else:
                    v_pred_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                    v_pred_label = call_model(train_state, x, t_vector, dt_base, labels)
                    v = v_pred_uncond + cfg_scale * (v_pred_label - v_pred_uncond)

                if FLAGS.model.train_type == 'consistency':
                    eps = shard_data(jax.random.normal(jax.random.fold_in(eps_key, ti), images_shape))
                    x1pred = x + v * (1-t)
                    x = x1pred * (t+delta_t) + eps * (1-t-delta_t)
                else:
                    x = x + v * delta_t # Euler sampling.
            x1.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
            lab.append(np.array(jax.experimental.multihost_utils.process_allgather(labels)))
            if FLAGS.model.use_stable_vae:
                x = vae_decode(x) # Image is in [-1, 1] space.
                if FLAGS.samples_dir is not None:
                    import cv2
                    import os
                    os.makedirs(FLAGS.samples_dir, exist_ok=True)
                    x_img = ((np.array(jax.experimental.multihost_utils.process_allgather(x)) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                    if jax.process_index() == 0:
                        global_batch_size = x_img.shape[0]
                        start_idx = fid_it * global_batch_size
                        for i, img in enumerate(x_img):
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(os.path.join(FLAGS.samples_dir, f"{start_idx + i:06d}.png"), img_bgr)
                elif num_generations < 10000:
                    x_render.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
            if get_fid_activations is not None:
                x_resized = jax.image.resize(x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
                x_resized = jnp.clip(x_resized, -1, 1)
                acts = get_fid_activations(x_resized)[..., 0, 0, :] # [devices, batch//devices, 2048]
                acts = jax.experimental.multihost_utils.process_allgather(acts)
                acts = np.array(acts)
                activations.append(acts)
        
        if jax.process_index() == 0 and get_fid_activations is not None:
            activations = np.concatenate(activations, axis=0)
            activations = activations.reshape((-1, activations.shape[-1]))
            mu1 = np.mean(activations, axis=0)
            sigma1 = np.cov(activations, rowvar=False)
            fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
            print(f"FID is {fid}")
            print(f"FID is {fid}")
            print(f"FID is {fid}")


            if FLAGS.save_dir is not None:
                os.makedirs(FLAGS.save_dir, exist_ok=True)
                x_render = np.concatenate(x_render, axis=0)
                np.save(FLAGS.save_dir + f'/x_render.npy', x_render)

                # x0 = np.concatenate(x0, axis=0)
                # x1 = np.concatenate(x1, axis=0)
                # lab = np.concatenate(lab, axis=0)
                # os.makedirs(FLAGS.save_dir, exist_ok=True)
                # np.save(FLAGS.save_dir + f'/x0.npy', x0)
                # np.save(FLAGS.save_dir + f'/x1.npy', x1)
                # np.save(FLAGS.save_dir + f'/lab.npy', lab)
