from .custom import CustomDataset
from .builder import DATASETS, PIPELINES
from pycocotools.coco import COCO

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
        file_client_args=dict(backend='disk'),
    ): 
        self.gt_seg_map_loader = CustomLoadAnnotations(coco_json_path)
        self.img_infos=gt_seg_map_loader.load_annotations()
    

    



class CustomLoadAnnotations(object):
    def __init__(
        self,
        coco_json_path,
        reduce_zero_label=False,
        file_client_args=dict(backend='disk'),
        imdecode_backend='pillow'):

        self.coco = COCO(coco_json_path)
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        coco_ind = results['ann_info']['coco_image_id']
        image_info = self.coco.loadImgs(coco_ind)[0]
        ann_inds = self.coco.getAnnIds(coco_ind)
        anns = self.coco.loadAnns(ann_inds)
        anns = list(sorted(anns, key=lambda x: -x['area']))

        gt_sematic_seg = np.zeros((image_info['heights'], image_info['width']))
        for ann in anns:
            gt_semantic_seg[self.coco.annToMask(ann)==1] = ann['category_id']
        gt_semantic_seg = gt_semantic_seg.astype(np.int64)

        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg==old_id] = new_id
        if self.reduce_zero_label:
            gt_semantic_seg[gt_semantic_seg==0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

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