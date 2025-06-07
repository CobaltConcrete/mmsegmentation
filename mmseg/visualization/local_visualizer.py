# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import time
import cv2
import mmcv
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.structures import PixelData
from mmengine.visualization import Visualizer

from mmseg.registry import VISUALIZERS
from mmseg.structures import SegDataSample
from mmseg.utils import get_classes, get_palette


@VISUALIZERS.register_module()
class SegLocalVisualizer(Visualizer):
    """Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        classes (list, optional): Input classes for result rendering, as the
            prediction of segmentation model is a segment map with label
            indices, `classes` is a list which includes items responding to the
            label indices. If classes is not defined, visualizer will take
            `cityscapes` classes by default. Defaults to None.
        palette (list, optional): Input palette for result rendering, which is
            a list of color palette responding to the classes. Defaults to None.
        dataset_name (str, optional): `Dataset name or alias <https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/class_names.py#L302-L317>`_
            visulizer will use the meta information of the dataset i.e. classes
            and palette, but the `classes` and `palette` have higher priority.
            Defaults to None.
        alpha (int, float): The transparency of segmentation mask.
                Defaults to 0.8.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import PixelData
        >>> from mmseg.structures import SegDataSample
        >>> from mmseg.visualization import SegLocalVisualizer

        >>> seg_local_visualizer = SegLocalVisualizer()
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_sem_seg_data = dict(data=torch.randint(0, 2, (1, 10, 12)))
        >>> gt_sem_seg = PixelData(**gt_sem_seg_data)
        >>> gt_seg_data_sample = SegDataSample()
        >>> gt_seg_data_sample.gt_sem_seg = gt_sem_seg
        >>> seg_local_visualizer.dataset_meta = dict(
        >>>     classes=('background', 'foreground'),
        >>>     palette=[[120, 120, 120], [6, 230, 230]])
        >>> seg_local_visualizer.add_datasample('visualizer_example',
        ...                         image, gt_seg_data_sample)
        >>> seg_local_visualizer.add_datasample(
        ...                        'visualizer_example', image,
        ...                         gt_seg_data_sample, show=True)
    """  # noqa

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 classes: Optional[List] = None,
                 palette: Optional[List] = None,
                 dataset_name: Optional[str] = None,
                 alpha: float = 0.8,
                 **kwargs):
        super().__init__(name, image, vis_backends, save_dir, **kwargs)
        self.alpha: float = alpha
        self.set_dataset_meta(palette, classes, dataset_name)

    def _get_center_loc(self, mask: np.ndarray) -> np.ndarray:
        """Get semantic seg center coordinate.

        Args:
            mask: np.ndarray: get from sem_seg
        """
        loc = np.argwhere(mask == 1)

        loc_sort = np.array(
            sorted(loc.tolist(), key=lambda row: (row[0], row[1])))
        y_list = loc_sort[:, 0]
        unique, indices, counts = np.unique(
            y_list, return_index=True, return_counts=True)
        y_loc = unique[counts.argmax()]
        y_most_freq_loc = loc[loc_sort[:, 0] == y_loc]
        center_num = len(y_most_freq_loc) // 2
        x = y_most_freq_loc[center_num][1]
        y = y_most_freq_loc[center_num][0]
        return np.array([x, y])

    def _draw_sem_seg(self,
                      image: np.ndarray,
                      sem_seg: PixelData,
                      classes: Optional[List],
                      palette: Optional[List],
                      with_labels: Optional[bool] = False) -> np.ndarray:
        """Draw semantic seg of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            sem_seg (:obj:`PixelData`): Data structure for pixel-level
                annotations or predictions.
            classes (list, optional): Input classes for result rendering, as
                the prediction of segmentation model is a segment map with
                label indices, `classes` is a list which includes items
                responding to the label indices. If classes is not defined,
                visualizer will take `cityscapes` classes by default.
                Defaults to None.
            palette (list, optional): Input palette for result rendering, which
                is a list of color palette responding to the classes.
                Defaults to None.
            with_labels(bool, optional): Add semantic labels in visualization
                result, Default to True.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        # classes += ('Unknown',)
        # if len(palette) < len(classes):
        #     palette.append([255, 0, 0])

        num_classes = len(classes)

        sem_seg = sem_seg.cpu().data
        ids = np.unique(sem_seg)[::-1]
        legal_indices = ids < num_classes
        ids = ids[legal_indices]
        labels = np.array(ids, dtype=np.int64)

        colors = [palette[label] for label in labels]

        mask = np.zeros_like(image, dtype=np.uint8)
        for label, color in zip(labels, colors):
            mask[sem_seg[0] == label, :] = color

        if False:
            font = cv2.FONT_HERSHEY_SIMPLEX
            # (0,1] to change the size of the text relative to the image
            scale = 0.05
            fontScale = min(image.shape[0], image.shape[1]) / (25 / scale)
            fontColor = (255, 255, 255)
            if image.shape[0] < 300 or image.shape[1] < 300:
                thickness = 1
                rectangleThickness = 1
            else:
                thickness = 2
                rectangleThickness = 2
            lineType = 2

            if isinstance(sem_seg[0], torch.Tensor):
                masks = sem_seg[0].numpy() == labels[:, None, None]
            else:
                masks = sem_seg[0] == labels[:, None, None]
            # print("MASK: ", masks)
            masks = masks.astype(np.uint8)
            for mask_num in range(len(labels)):
                classes_id = labels[mask_num]
                classes_color = colors[mask_num]
                loc = self._get_center_loc(masks[mask_num])
                text = classes[classes_id]
                (label_width, label_height), baseline = cv2.getTextSize(
                    text, font, fontScale, thickness)
                mask = cv2.rectangle(mask, loc,
                                     (loc[0] + label_width + baseline,
                                      loc[1] + label_height + baseline),
                                     classes_color, -1)
                mask = cv2.rectangle(mask, loc,
                                     (loc[0] + label_width + baseline,
                                      loc[1] + label_height + baseline),
                                     (0, 0, 0), rectangleThickness)
                mask = cv2.putText(mask, text, (loc[0], loc[1] + label_height),
                                   font, fontScale, fontColor, thickness,
                                   lineType)
                
        color_seg = (image * (1 - self.alpha) + mask * self.alpha).astype(
            np.uint8)
        self.set_image(color_seg)
        return color_seg
    
    import time
    
    def _draw_sem_seg_GPU(self,
                        image: np.ndarray,
                        sem_seg: PixelData,
                        classes: Optional[List],
                        palette: Optional[List],
                        with_labels: Optional[bool] = True,
                        print_timings: bool = True) -> np.ndarray:
        """Draw semantic segmentation of GT or prediction using GPU where possible.

        Args:
            image (np.ndarray): The image to draw.
            sem_seg (:obj:`PixelData`): Data structure for pixel-level annotations or predictions.
            classes (list, optional): Input classes for result rendering.
            palette (list, optional): Input palette for result rendering.
            with_labels(bool, optional): Add semantic labels in visualization result.
            print_timings (bool): Flag to print timings for various steps.

        Returns:
            np.ndarray: The drawn image which channel is RGB.
        """
        
        # Ensure sem_seg is a PyTorch tensor, and move it to the right device (GPU if available)
        if isinstance(sem_seg, torch.Tensor):
            device = sem_seg.device
        else:
            if hasattr(sem_seg, 'data'):  # Custom PixelData with 'data' attribute
                sem_seg = torch.tensor(sem_seg.data, dtype=torch.long)
            else:
                raise ValueError("PixelData does not have a 'data' attribute or is not in an expected format.")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move the image to the same device as sem_seg for GPU computation
        image = torch.tensor(image).to(device)

        if print_timings:
            print(f"Using device: {device}")

        # Timer for the initial setup (getting the number of classes and unique IDs)
        start_time = time.time()
        num_classes = len(classes)
        ids = torch.unique(sem_seg)  # Get unique values from the tensor
        ids = torch.flip(ids, dims=[0])  # Flip (reverse) the tensor along the first axis
        ids = ids.to(device)  # Move the reversed tensor to the device (GPU or CPU)
        
        legal_indices = ids < num_classes  # Ensure IDs are valid
        ids = ids[legal_indices]
        labels = ids.to(torch.int64)

        # Map labels to colors based on palette
        colors = [palette[label] for label in labels]
        mask = torch.zeros_like(image, dtype=torch.uint8, device=device)  # Create a mask on the same device
        setup_time = time.time() - start_time
        if print_timings:
            print(f"Initial setup time: {setup_time:.4f} seconds")

        # Timer for drawing the segmentation mask
        start_time = time.time()
        for label, color in zip(labels, colors):
            mask[sem_seg[0] == label] = torch.tensor(color, dtype=torch.uint8, device=device)
        drawing_mask_time = time.time() - start_time
        if print_timings:
            print(f"Mask drawing time: {drawing_mask_time:.4f} seconds")

        # Timer for adding labels (if enabled)
        label_drawing_time = 0
        if with_labels:
            start_time = time.time()
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.05
            fontScale = min(image.shape[0], image.shape[1]) / (25 / scale)
            fontColor = (255, 255, 255)
            thickness = 2 if image.shape[0] >= 300 and image.shape[1] >= 300 else 1
            lineType = 2

            # Use CPU for label drawing and text rendering to avoid unnecessary GPU overhead
            masks = [(sem_seg[0] == label).cpu() for label in labels]  # Move mask computations to CPU
            for mask_num, mask_for_label in enumerate(masks):
                classes_id = labels[mask_num]
                classes_color = colors[mask_num]
                loc = self._get_center_loc(mask_for_label.numpy())  # Convert to CPU for OpenCV
                text = classes[classes_id]

                # Get text size and draw label
                (label_width, label_height), baseline = cv2.getTextSize(text, font, fontScale, thickness)
                mask_numpy = mask_for_label.numpy().astype(np.uint8)  # Convert to NumPy for OpenCV
                mask_numpy = cv2.rectangle(mask_numpy, loc,
                                        (loc[0] + label_width + baseline, loc[1] + label_height + baseline),
                                        classes_color, -1)
                mask_numpy = cv2.rectangle(mask_numpy, loc,
                                        (loc[0] + label_width + baseline, loc[1] + label_height + baseline),
                                        (0, 0, 0), thickness)  # Add border
                mask_numpy = cv2.putText(mask_numpy, text, (loc[0], loc[1] + label_height),
                                        font, fontScale, fontColor, thickness, lineType)

                # Convert back to tensor after processing (if necessary)
                mask[0] = torch.from_numpy(mask_numpy).to(device)

            label_drawing_time = time.time() - start_time
            if print_timings:
                print(f"Label drawing time: {label_drawing_time:.4f} seconds")

        # Timer for blending the segmentation mask with the original image
        start_time = time.time()
        color_seg = (image * (1 - self.alpha) + mask * self.alpha).to(torch.uint8)
        blending_time = time.time() - start_time
        if print_timings:
            print(f"Blending time: {blending_time:.4f} seconds")

        # Convert to numpy array before passing it to OpenCV
        color_seg_np = color_seg.cpu().numpy()

        # Timer for setting the image (if any additional processing is done here)
        start_time = time.time()
        self.set_image(color_seg_np)  # Now it's a NumPy array
        set_image_time = time.time() - start_time
        if print_timings:
            print(f"Set image time: {set_image_time:.4f} seconds")

        # Total time for the entire method
        total_time = (setup_time + drawing_mask_time + label_drawing_time + 
                    blending_time + set_image_time)
        if print_timings:
            print(f"Total time for _draw_sem_seg_GPU: {total_time:.4f} seconds")

        return color_seg_np  # Return as NumPy array for display or further processing


    def _draw_depth_map(self, image: np.ndarray,
                        depth_map: PixelData) -> np.ndarray:
        """Draws a depth map on a given image.

        This function takes an image and a depth map as input,
        renders the depth map, and concatenates it with the original image.
        Finally, it updates the internal image state of the visualizer with
        the concatenated result.

        Args:
            image (np.ndarray): The original image where the depth map will
                be drawn. The array should be in the format HxWx3 where H is
                the height, W is the width.

            depth_map (PixelData): Depth map to be drawn. The depth map
                should be in the form of a PixelData object. It will be
                converted to a torch tensor if it is a numpy array.

        Returns:
            np.ndarray: The concatenated image with the depth map drawn.

        Example:
            >>> depth_map_data = PixelData(data=torch.rand(1, 10, 10))
            >>> image = np.random.randint(0, 256,
            >>>                           size=(10, 10, 3)).astype('uint8')
            >>> visualizer = SegLocalVisualizer()
            >>> visualizer._draw_depth_map(image, depth_map_data)
        """
        depth_map = depth_map.cpu().data
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map)
        if depth_map.ndim == 2:
            depth_map = depth_map[None]

        depth_map = self.draw_featmap(depth_map, resize_shape=image.shape[:2])
        out_image = np.concatenate((image, depth_map), axis=0)
        self.set_image(out_image)
        return out_image

    def set_dataset_meta(self,
                         classes: Optional[List] = None,
                         palette: Optional[List] = None,
                         dataset_name: Optional[str] = None) -> None:
        """Set meta information to visualizer.

        Args:
            classes (list, optional): Input classes for result rendering, as
                the prediction of segmentation model is a segment map with
                label indices, `classes` is a list which includes items
                responding to the label indices. If classes is not defined,
                visualizer will take `cityscapes` classes by default.
                Defaults to None.
            palette (list, optional): Input palette for result rendering, which
                is a list of color palette responding to the classes.
                Defaults to None.
            dataset_name (str, optional): `Dataset name or alias <https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/class_names.py#L302-L317>`_
                visulizer will use the meta information of the dataset i.e.
                classes and palette, but the `classes` and `palette` have
                higher priority. Defaults to None.
        """  # noqa
        # Set default value. When calling
        # `SegLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        if dataset_name is None:
            dataset_name = 'cityscapes'
        classes = classes if classes else get_classes(dataset_name)
        palette = palette if palette else get_palette(dataset_name)
        assert len(classes) == len(
            palette), 'The length of classes should be equal to palette'
        self.dataset_meta: dict = {'classes': classes, 'palette': palette}

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional[SegDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            out_file: Optional[str] = None,
            step: int = 0,
            with_labels: Optional[bool] = True,
            print_timings: bool = False
        ) -> None:
        """Draw datasample and save to all backends."""
        
        # Start the timer for dataset meta retrieval (getting classes/palette)
        start_time = time.time()
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)
        meta_time = time.time() - start_time
        if print_timings:
            print(f"Dataset meta retrieval time: {meta_time:.4f} seconds")

        # Initialize all time variables
        gt_drawing_time = 0
        gt_depth_time = 0
        pred_drawing_time = 0
        pred_depth_time = 0
        concat_time = 0
        show_time = 0
        save_time = 0
        add_image_time = 0

        gt_img_data = None
        pred_img_data = None

        # Start timer for drawing ground truth
        if draw_gt and data_sample is not None:
            if 'gt_sem_seg' in data_sample:
                assert classes is not None, 'class information is not provided when visualizing semantic segmentation results.'
                start_time = time.time()
                gt_img_data = self._draw_sem_seg_GPU(image, data_sample.gt_sem_seg, classes, palette, with_labels=True)
                gt_drawing_time = time.time() - start_time
                if print_timings:
                    print(f"Ground truth drawing time: {gt_drawing_time:.4f} seconds")

            if 'gt_depth_map' in data_sample:
                gt_img_data = gt_img_data if gt_img_data is not None else image
                start_time = time.time()
                gt_img_data = self._draw_depth_map(gt_img_data, data_sample.gt_depth_map)
                gt_depth_time = time.time() - start_time
                if print_timings:
                    print(f"Ground truth depth map drawing time: {gt_depth_time:.4f} seconds")

        # Start timer for drawing prediction
        if draw_pred and data_sample is not None:
            if 'pred_sem_seg' in data_sample:
                assert classes is not None, 'class information is not provided when visualizing semantic segmentation results.'
                start_time = time.time()
                pred_img_data = self._draw_sem_seg(image, data_sample.pred_sem_seg, classes, palette, with_labels=True)
                pred_drawing_time = time.time() - start_time
                if print_timings:
                    print(f"Prediction drawing time: {pred_drawing_time:.4f} seconds")

            if 'pred_depth_map' in data_sample:
                pred_img_data = pred_img_data if pred_img_data is not None else image
                start_time = time.time()
                pred_img_data = self._draw_depth_map(pred_img_data, data_sample.pred_depth_map)
                pred_depth_time = time.time() - start_time
                if print_timings:
                    print(f"Prediction depth map drawing time: {pred_depth_time:.4f} seconds")

        # Timer for concatenating the images
        start_time = time.time()
        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data
        concat_time = time.time() - start_time
        if print_timings:
            print(f"Image concatenation time: {concat_time:.4f} seconds")

        # Timer for showing the image
        if show:
            start_time = time.time()
            self.show(drawn_img, win_name=name, wait_time=wait_time)
            show_time = time.time() - start_time
            if print_timings:
                print(f"Show image time: {show_time:.4f} seconds")

        # Timer for saving the image (if needed)
        if out_file is not None:
            start_time = time.time()
            mmcv.imwrite(mmcv.rgb2bgr(drawn_img), out_file)
            save_time = time.time() - start_time
            if print_timings:
                print(f"Saving image time: {save_time:.4f} seconds")
        else:
            start_time = time.time()
            self.add_image(name, drawn_img, step)
            add_image_time = time.time() - start_time
            if print_timings:
                print(f"Add image time: {add_image_time:.4f} seconds")

        # Total time for the entire function
        total_time = (meta_time + gt_drawing_time + gt_depth_time + 
                    pred_drawing_time + pred_depth_time + concat_time + 
                    show_time + save_time + add_image_time)
        if print_timings:
            print(f"Total time for add_datasample: {total_time:.4f} seconds")
