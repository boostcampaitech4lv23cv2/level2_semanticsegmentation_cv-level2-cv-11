from .custom import CustomDataset
from .builder import DATASETS

@DATASETS.register_module()
class TrashDataset(CustomDataset):

    CLASSES = (
    'Backgroud',
    'General trash',
    'Paper',
    'Paper pack',
    'Metal',
    'Glass',
    'Plastic',
    'Styrofoam',
    'Plastic bag',
    'Battery',
    'Clothing'
)
    PALETTE = [[0, 0, 0], [192, 0, 128], [0, 128, 192], 
    [0, 128, 64], [128, 0, 0], [64, 0, 128], [64, 0, 192],
    [192, 128, 64], [192, 192, 128], [64, 64, 128], [128, 0, 192]]
    

    def __init__(
        self,
        pipeline,
        coco_json_path,
        img_dir,
        is_valid,
        img_suffix='.jpg',
        ann_dir=None,
        seg_map_suffix='.png',
        split=None,
        data_root=None,
        test_mode=False,
        ignore_index=255,
        reduce_zero_label=False,
        classes=None,
        palette=None,
        gt_seg_map_loader_cfg=None,
        file_client_args=dict(backend='disk')
    ):
    self.gt_seg_map_loader = CustomLoadAnnotations(coco_json_path)


    def load_annotations(self):
        img_infos = []

        for img in self.coco.imgs.values():
            img_info = dict(filename=img['file_name'])
            img_infos.append(img_info)

            img_info['ann'] = dict(coco_img_id=img['id'])
        return img_infos

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        result['img_prefix'] = self.img_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    
    


    