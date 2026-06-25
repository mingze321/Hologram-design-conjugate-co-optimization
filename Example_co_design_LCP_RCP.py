import os
import sys
from pathlib import Path


def relaunch_with_nvidia_libs():
    if sys.platform != 'linux' or os.environ.get('JAX_NVIDIA_LIBS_READY') == '1':
        return

    lib_dirs = sorted(Path(sys.prefix).glob('lib/python*/site-packages/nvidia/*/lib'))
    if not lib_dirs:
        return

    lib_paths = [str(path) for path in lib_dirs]
    current_ld_paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
    missing_paths = [path for path in lib_paths if path not in current_ld_paths]
    if not missing_paths:
        return

    env = os.environ.copy()
    env['JAX_NVIDIA_LIBS_READY'] = '1'
    env['LD_LIBRARY_PATH'] = ':'.join(missing_paths + current_ld_paths)
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


relaunch_with_nvidia_libs()

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.io import loadmat


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUT_DIR = BASE_DIR / 'out_dual'
OUT_DIR.mkdir(exist_ok=True)

FILTER_SIZE = 807
APERTURE_SIZE = 800
FULL_SIZE = 2048
RESOLUTION = 1.65 * 2
WAVELENGTH = 4.0
K_CUTOFF = 0.4
DISTANCE1 = FILTER_SIZE*RESOLUTION * 2.0
PADDING = ((620, 621), (620, 621))


def data_path(filename):
    return str(DATA_DIR / filename)


def out_path(filename):
    return str(OUT_DIR / filename)


def save_matshow(array, title, filename, transform=np.abs):
    plt.matshow(transform(np.asarray(array)), aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.savefig(out_path(filename), dpi=300)
    plt.close()


def normalize_target_power(target_x, target_y):
    target_power = float(APERTURE_SIZE * APERTURE_SIZE)
    current_power = np.sum(np.abs(target_x) ** 2 + np.abs(target_y) ** 2)

    if current_power <= 0:
        raise ValueError('Target power is zero; cannot normalize target fields.')

    scale = np.sqrt(target_power / current_power)
    print(
        'Target power normalized from {} to {} with amplitude scale {}'.format(
            current_power,
            target_power,
            scale,
        )
    )
    return target_x * scale, target_y * scale


def set_target():
    target_x = np.transpose(loadmat(data_path('CUNY_logo.mat'))['E_target_x'].reshape(FULL_SIZE, FULL_SIZE))
    target_y = np.transpose(loadmat(data_path('ASRC_logo.mat'))['E_target_x'].reshape(FULL_SIZE, FULL_SIZE))
    target_x, target_y = normalize_target_power(target_x, target_y)

    save_matshow(target_x, 'target image intensity x ', 'target intensity x.png', np.abs)
    save_matshow(target_x, 'target image phase x', 'target phase x.png', np.angle)
    save_matshow(target_y, 'target image intensity y', 'target intensity y.png', np.abs)
    save_matshow(target_y, 'target image phase y', 'target phase y.png', np.angle)

    return (
        jnp.asarray(target_x, dtype=jnp.complex128),
        jnp.asarray(target_y, dtype=jnp.complex128),
    )


def aperture_np(xcord, ycord, side_length):
    half_width = side_length / 2
    return (
        (xcord >= -half_width)
        & (xcord < half_width)
        & (ycord >= -half_width)
        & (ycord < half_width)
    ).astype(np.float64)


def def_lens_as_starting():
    cord = np.arange(-FILTER_SIZE / 2, FILTER_SIZE / 2) * RESOLUTION
    xx_propagation, yy_propagation = np.meshgrid(cord, cord)
    f = 2 * RESOLUTION * FILTER_SIZE
    mesh = np.sqrt(xx_propagation ** 2 + yy_propagation ** 2)
    return (2 * 3.14 / WAVELENGTH) * np.sqrt(mesh ** 2 + f ** 2)


def build_static_arrays():
    cord = np.arange(-FULL_SIZE / 2, FULL_SIZE / 2)
    xx_propagation, yy_propagation = np.meshgrid(cord, cord)

    aperture_filter = aperture_np(xx_propagation, yy_propagation, APERTURE_SIZE)

    kfrequency = 1 / FULL_SIZE / RESOLUTION * WAVELENGTH
    kx = xx_propagation * kfrequency
    ky = yy_propagation * kfrequency
    field_stop_k = ((kx ** 2 + ky ** 2) < K_CUTOFF ** 2).astype(np.float64)
    phase_scale = 2 * np.pi * DISTANCE1 / WAVELENGTH
    transfer_function = (
        np.exp(1j * phase_scale * np.sqrt(1 - kx ** 2 - ky ** 2 + 0j))
        * field_stop_k
    )
    transfer_function = np.fft.ifftshift(transfer_function)

    return (
        jnp.asarray(aperture_filter, dtype=jnp.complex128),
        jnp.asarray(transfer_function, dtype=jnp.complex128),
    )


def forward_propagation(plane, transfer_function):
    spectrum_layerout = jnp.fft.fft2(plane)
    return jnp.fft.ifft2(spectrum_layerout * transfer_function)


def model_apply(filter1_phase_x, aperture_filter, transfer_function):
    filter1_x_c = jnp.exp(1j * filter1_phase_x)
    filter1_y_c = jnp.exp(-1j * filter1_phase_x)

    filter1_x_full = jnp.pad(filter1_x_c, PADDING) * aperture_filter
    filter1_y_full = jnp.pad(filter1_y_c, PADDING) * aperture_filter

    field_pre2_x = forward_propagation(filter1_x_full, transfer_function)
    field_pre2_y = forward_propagation(filter1_y_full, transfer_function)

    return field_pre2_x, field_pre2_y


def make_loss_fn(target_x, target_y, static_arrays):
    aperture_filter, transfer_function = static_arrays
    target_x_intensity = jnp.abs(target_x) ** 2
    target_y_intensity = jnp.abs(target_y) ** 2
    input_power = 2.0 * jnp.sum(jnp.abs(aperture_filter) ** 2)

    def loss_fn(filter1_phase_x):
        output_x, output_y = model_apply(
            filter1_phase_x,
            aperture_filter,
            transfer_function,
        )
        output_x_intensity = jnp.abs(output_x) ** 2
        output_y_intensity = jnp.abs(output_y) ** 2
        loss_x = jnp.mean((output_x_intensity - target_x_intensity) ** 2)
        loss_y = jnp.mean((output_y_intensity - target_y_intensity) ** 2)
        designed_efficiency = jnp.sum(output_x_intensity + output_y_intensity) / input_power
        return loss_x + loss_y, (loss_x, loss_y, designed_efficiency)

    return loss_fn


def save_filter(filter1_phase_x, epoch):
    filter1_x_c = jnp.exp(1j * filter1_phase_x)

    np.savetxt(
        out_path('filter1_x_epoch_' + str(epoch) + '.csv'),
        np.asarray(jnp.angle(filter1_x_c)),
        delimiter=',',
    )



def save_checkpoint_images(filter1_phase_x, static_arrays, epoch):
    aperture_filter, transfer_function = static_arrays
    output_x, output_y = model_apply(
        filter1_phase_x,
        aperture_filter,
        transfer_function,
    )
    output_x = np.asarray(jnp.abs(output_x[512 + 200:1536 - 200, 512 + 200:1536 - 200]) ** 2)
    output_y = np.asarray(jnp.abs(output_y[512 + 200:1536 - 200, 512 + 200:1536 - 200]) ** 2)

    save_matshow(
        output_x,
        'designed image intensity',
        'far_field designed image intensity_xpol' + str(epoch) + '.png',
        lambda x: x,
    )
    save_matshow(
        output_y,
        'designed image intensity',
        'far_field designed image intensity_ypol' + str(epoch) + '.png',
        lambda x: x,
    )


def main():
    print('JAX', jax.__version__)
    print(jax.devices())

    target_x, target_y = set_target()
    static_arrays = build_static_arrays()

    filter1_phase_x = jnp.asarray(def_lens_as_starting(), dtype=jnp.float64)
    loss_fn = make_loss_fn(target_x, target_y, static_arrays)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(filter1_phase_x)

    @jax.jit
    def train_step(params, opt_state):
        (loss, aux), grads = value_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    epochs = int(os.environ.get('EPOCHS1', '120000'))
    print('start training')

    for epoch in range(epochs):
        filter1_phase_x, opt_state, loss, aux = train_step(filter1_phase_x, opt_state)
        loss_x, loss_y, designed_efficiency = aux

        if epoch % 150 == 0:
            save_filter(filter1_phase_x, epoch)
            save_checkpoint_images(filter1_phase_x, static_arrays, epoch)
            print(
                'Epoch {}, Loss: {}, loss_x: {}, loss_y: {}, designed efficiency: {}'.format(
                    epoch + 1,
                    float(loss),
                    float(loss_x),
                    float(loss_y),
                    float(designed_efficiency),
                )
            )


if __name__ == '__main__':
    main()
