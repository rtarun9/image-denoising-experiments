
import numpy as np

from scipy.interpolate import SmoothBivariateSpline as SBS
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure


class EMD2D:
    def __init__(self, **config):
        # ProtoIMF related
        self.mse_thr = 0.01
        self.mean_thr = 0.01

        self.FIXE = 0
        self.FIXE_H = 0

        self.MAX_ITERATION = 1000

        # Update based on options
        for key in config.keys():
            if key in self.__dict__.keys():
                self.__dict__[key] = config[key]

    def __call__(self, image, max_imf=-1):
        return self.emd(image, max_imf=max_imf)

    def extract_max_min_spline(self, image):
        big_image = self.prepare_image(image)
        big_min_peaks, big_max_peaks = self.find_extrema(big_image)

        # Prepare grid for interpolation. Doesn't seem necessary.
        xi = np.arange(image.shape[0], image.shape[0] * 2)
        yi = np.arange(image.shape[1], image.shape[1] * 2)

        big_min_image_val = big_image[big_min_peaks]
        big_max_image_val = big_image[big_max_peaks]
        min_env = self.spline_points(big_min_peaks[0], big_min_peaks[1], big_min_image_val, xi, yi)
        max_env = self.spline_points(big_max_peaks[0], big_max_peaks[1], big_max_image_val, xi, yi)

        return min_env, max_env

    @classmethod
    def prepare_image(cls, image):
        shape = image.shape
        big_image = np.zeros((shape[0] * 3, shape[1] * 3))

        image_lr = np.fliplr(image)
        image_ud = np.flipud(image)
        image_ud_lr = np.flipud(image_lr)
        image_lr_ud = np.fliplr(image_ud)

        # Fill center with default image
        big_image[shape[0] : 2 * shape[0], shape[1] : 2 * shape[1]] = image

        # Fill left center
        big_image[shape[0] : 2 * shape[0], : shape[1]] = image_lr

        # Fill right center
        big_image[shape[0] : 2 * shape[0], 2 * shape[1] :] = image_lr

        # Fill center top
        big_image[: shape[0], shape[1] : shape[1] * 2] = image_ud

        # Fill center bottom
        big_image[2 * shape[0] :, shape[1] : 2 * shape[1]] = image_ud

        # Fill left top
        big_image[: shape[0], : shape[1]] = image_ud_lr

        # Fill left bottom
        big_image[2 * shape[0] :, : shape[1]] = image_ud_lr

        # Fill right top
        big_image[: shape[0], 2 * shape[1] :] = image_lr_ud

        # Fill right bottom
        big_image[2 * shape[0] :, 2 * shape[1] :] = image_lr_ud

        return big_image

    @classmethod
    def spline_points(cls, X, Y, Z, xi, yi):
        """Interpolates for given set of points"""

        spline = SBS(X, Y, Z)

        return spline(xi, yi)

    @classmethod
    def find_extrema(cls, image):
        # define an 3x3 neighborhood
        neighborhood = generate_binary_structure(2, 2)

        # apply the local maximum filter; all pixel of maximal value
        # in their neighborhood are set to 1
        local_min = maximum_filter(-image, footprint=neighborhood) == -image
        local_max = maximum_filter(image, footprint=neighborhood) == image

        # can't distinguish between background zero and filter zero
        background = image == 0

        # appear along the bg border (artifact of the local max filter)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

        # we obtain the final mask, containing only peaks,
        # by removing the background from the local_max mask (xor operation)
        min_peaks = local_min ^ eroded_background
        max_peaks = local_max ^ eroded_background

        min_peaks = local_min
        max_peaks = local_max
        min_peaks[[0, -1], :] = False
        min_peaks[:, [0, -1]] = False
        max_peaks[[0, -1], :] = False
        max_peaks[:, [0, -1]] = False

        min_peaks = np.nonzero(min_peaks)
        max_peaks = np.nonzero(max_peaks)

        return min_peaks, max_peaks

    @classmethod
    def end_condition(cls, image, IMFs):
        rec = np.sum(IMFs, axis=0)

        # If reconstruction is perfect, no need for more tests
        if np.allclose(image, rec):
            return True

        return False

    def check_proto_imf(self, proto_imf, proto_imf_prev, mean_env):
        if np.all(np.abs(mean_env - mean_env.mean()) < self.mean_thr):
            # if np.all(np.abs(mean_env)<self.mean_thr):
            return True

        # If very little change with sifting
        if np.allclose(proto_imf, proto_imf_prev):
            return True

        # If IMF mean close to zero (below threshold)
        if np.mean(np.abs(proto_imf)) < self.mean_thr:
            return True

        # Everything relatively close to 0
        mse_proto_imf = np.mean(proto_imf * proto_imf)
        if mse_proto_imf < self.mse_thr:
            return True

        return False

    def emd(self, image, max_imf=-1):
        image_min, image_max = np.min(image), np.max(image)
        offset = image_min
        scale = image_max - image_min

        image_s = (image - offset) / scale

        imf = np.zeros(image.shape)
        imf_old = imf.copy()

        imfNo = 0
        IMF = np.empty((imfNo,) + image.shape)
        notFinished = True

        while notFinished:

            res = image_s - np.sum(IMF[:imfNo], axis=0)
            imf = res.copy()
            mean_env = np.zeros(image.shape)
            stop_sifting = False

            # Counters
            n = 0  # All iterations for current imf.
            n_h = 0  # counts when mean(proto_imf) < threshold

            while not stop_sifting and n < self.MAX_ITERATION:
                n += 1

                min_peaks, max_peaks = self.find_extrema(imf)

                if len(min_peaks[0]) > 4 and len(max_peaks[0]) > 4:
                    imf_old = imf.copy()
                    imf = imf - mean_env

                    min_env, max_env = self.extract_max_min_spline(imf)

                    mean_env = 0.5 * (min_env + max_env)

                    imf_old = imf.copy()
                    imf = imf - mean_env

                    # Fix number of iterations
                    if self.FIXE:
                        if n >= self.FIXE + 1:
                            stop_sifting = True

                    # Fix number of iterations after number of zero-crossings
                    # and extrema differ at most by one.
                    elif self.FIXE_H:
                        if n == 1:
                            continue
                        if self.check_proto_imf(imf, imf_old, mean_env):
                            n_h += 1
                        else:
                            n_h = 0

                        # STOP if enough n_h
                        if n_h >= self.FIXE_H:
                            stop_sifting = True

                    # Stops after default stopping criteria are met
                    else:
                        if self.check_proto_imf(imf, imf_old, mean_env):
                            stop_sifting = True

                else:
                    notFinished = False
                    stop_sifting = True

            IMF = np.vstack((IMF, imf.copy()[None, :]))
            imfNo += 1

            if self.end_condition(image, IMF) or (max_imf > 0 and imfNo >= max_imf):
                notFinished = False
                break

        res = image_s - np.sum(IMF[:imfNo], axis=0)
        if not np.allclose(res, 0):
            IMF = np.vstack((IMF, res[None, :]))
            imfNo += 1

        IMF = IMF * scale
        IMF[-1] += offset
        return IMF

