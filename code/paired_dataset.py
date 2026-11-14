"""New helper for confirmed revision preprocessing, not recovered historical training."""
import csv
import SimpleITK as sitk
from torch.utils.data import Dataset
from preprocessing import prepare
class PairedVolumes(Dataset):
    def __init__(self,manifest,role):
        if role not in ('train','validation'):raise ValueError('Training roles only')
        with open(manifest,encoding='utf-8-sig',newline='') as f:self.rows=[r for r in csv.DictReader(f) if r['role']==role]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        return tuple(prepare(sitk.GetArrayFromImage(sitk.ReadImage(r[k])))[0] for k in ('input','target'))
