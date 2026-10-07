import cv2
import numpy as np
from typing import Tuple, List, Optional
from scipy.ndimage import median_filter


class MultiScalePointRefiner:

    def __init__(self,
                 scales: List[int] = None,
                 max_drift_iter: int = 10,
                 contrast_window: int = 3,
                 use_filter: bool = True,
                 filter_size: int = 3,
                 max_drift_shift: int = 1):
        self.scales = scales if scales is not None else [9, 7, 5, 3]
        self.max_drift_iter = max_drift_iter
        self.contrast_window = contrast_window
        self.use_filter = use_filter
        self.filter_size = filter_size
        self.max_drift_shift = max_drift_shift

    def refine(self,
               image: np.ndarray,
               coarse_point: Tuple[int, int],
               verbose: bool = False) -> Tuple[int, int]:

        if len(image.shape) == 3:

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        x, y = coarse_point
        h, w = gray.shape


        if x < 2 or x >= w - 2 or y < 2 or y >= h - 2:
            if verbose:
                print(f"[Refiner] Point on boundary, returning original: ({x}, {y})")
            return (x, y)


        if self.use_filter:
            filtered_image = median_filter(gray, size=self.filter_size)
        else:
            filtered_image = gray

        if verbose:
            print(f"[Refiner] Original point: ({x}, {y})")


        current_x, current_y = x, y

        for scale in self.scales:
            best_x, best_y = self._multiscale_search(
                filtered_image, current_x, current_y, scale
            )
            if verbose:
                print(f"[Refiner] Scale {scale}: ({best_x}, {best_y})")
            current_x, current_y = best_x, best_y


        peak_x, peak_y = current_x, current_y


        drifted_x, drifted_y = self._iterative_drift(
            filtered_image, current_x, current_y
        )
        if verbose:
            print(f"[Refiner] After drift: ({drifted_x}, {drifted_y})")


        final_x, final_y = self._validate_drift(
            filtered_image, drifted_x, drifted_y, (peak_x, peak_y)
        )
        if verbose:
            print(f"[Refiner] Final point: ({final_x}, {final_y})")

        return (final_x, final_y)

    def _multiscale_search(self,
                          image: np.ndarray,
                          x: int,
                          y: int,
                          window_size: int) -> Tuple[int, int]:
        h, w = image.shape
        half = window_size // 2


        x_min = max(0, x - half)
        x_max = min(w, x + half + 1)
        y_min = max(0, y - half)
        y_max = min(h, y + half + 1)

        best_score = -1
        best_x, best_y = x, y
        best_dist2 = float("inf")

        for cy in range(y_min, y_max):
            for cx in range(x_min, x_max):

                contrast = self._compute_local_contrast(image, cx, cy)

                dist2 = (cx - x) ** 2 + (cy - y) ** 2
                if contrast > best_score or (contrast == best_score
                                             and dist2 < best_dist2):
                    best_score = contrast
                    best_x, best_y = cx, cy
                    best_dist2 = dist2

        return (best_x, best_y)

    def _iterative_drift(self,
                        image: np.ndarray,
                        x: int,
                        y: int) -> Tuple[int, int]:
        current_x, current_y = x, y
        h, w = image.shape

        for _ in range(self.max_drift_iter):

            x_min = max(0, current_x - 2)
            x_max = min(w, current_x + 3)
            y_min = max(0, current_y - 2)
            y_max = min(h, current_y + 3)

            patch = image[y_min:y_max, x_min:x_max]

            if patch.size == 0:
                break


            total_weight = patch.sum()
            if total_weight == 0:
                break

            yy, xx = np.meshgrid(range(patch.shape[0]), range(patch.shape[1]), indexing='ij')
            weighted_x = (xx * patch).sum() / total_weight
            weighted_y = (yy * patch).sum() / total_weight

            new_x = x_min + weighted_x
            new_y = y_min + weighted_y


            if abs(new_x - current_x) < 0.5 and abs(new_y - current_y) < 0.5:
                break

            current_x = int(round(new_x))
            current_y = int(round(new_y))

        return (current_x, current_y)

    def _contrast_refine(self,
                        image: np.ndarray,
                        x: int,
                        y: int) -> Tuple[int, int]:
        return self._multiscale_search(image, x, y, self.contrast_window)

    def _validate_drift(self,
                        image: np.ndarray,
                        x: int,
                        y: int,
                        peak_xy: Tuple[int, int]) -> Tuple[int, int]:

        if self.max_drift_shift < 0:
            return (x, y)

        shift = max(abs(x - peak_xy[0]), abs(y - peak_xy[1]))
        if shift <= self.max_drift_shift:
            return (x, y)

        return self._contrast_refine(image, x, y)

    def _compute_local_contrast(self,
                                image: np.ndarray,
                                cx: int,
                                cy: int) -> float:
        h, w = image.shape


        x_min = max(0, cx - 1)
        x_max = min(w, cx + 2)
        y_min = max(0, cy - 1)
        y_max = min(h, cy + 2)

        local_patch = image[y_min:y_max, x_min:x_max]

        center_val = image[cy, cx]
        surround_mean = (local_patch.sum() - center_val) / (local_patch.size - 1 + 1e-8)


        contrast = center_val / (surround_mean + 1e-8)

        return contrast

    def refine_batch(self,
                     images: List[np.ndarray],
                     coarse_points: List[Tuple[int, int]],
                     verbose: bool = False) -> List[Tuple[int, int]]:
        refined_points = []
        for i, (img, pt) in enumerate(zip(images, coarse_points)):
            refined = self.refine(img, pt, verbose and i == 0)
            refined_points.append(refined)
        return refined_points

    def get_correction_statistics(self,
                                  images: List[np.ndarray],
                                  coarse_points: List[Tuple[int, int]],
                                  ground_truth_points: Optional[List[Tuple[int, int]]] = None
                                  ) -> dict:
        shifts = []
        corrected_count = 0

        for i, (img, pt) in enumerate(zip(images, coarse_points)):
            refined = self.refine(img, pt, verbose=False)
            shift = abs(refined[0] - pt[0]) + abs(refined[1] - pt[1])
            shifts.append(shift)
            if shift > 0:
                corrected_count += 1

        stats = {
            'mean_shift': np.mean(shifts),
            'max_shift': np.max(shifts),
            'std_shift': np.std(shifts),
            'corrected_ratio': corrected_count / len(coarse_points) if coarse_points else 0,
            'shifts': shifts
        }


        if ground_truth_points is not None:
            dist_to_gt_original = []
            dist_to_gt_refined = []
            for i, (img, pt) in enumerate(zip(images, coarse_points)):
                refined = self.refine(img, pt, verbose=False)
                gt = ground_truth_points[i]

                orig_dist = abs(pt[0] - gt[0]) + abs(pt[1] - gt[1])
                ref_dist = abs(refined[0] - gt[0]) + abs(refined[1] - gt[1])
                dist_to_gt_original.append(orig_dist)
                dist_to_gt_refined.append(ref_dist)

            stats['mean_dist_to_gt_original'] = np.mean(dist_to_gt_original)
            stats['mean_dist_to_gt_refined'] = np.mean(dist_to_gt_refined)
            stats['improvement'] = stats['mean_dist_to_gt_original'] - stats['mean_dist_to_gt_refined']

        return stats
