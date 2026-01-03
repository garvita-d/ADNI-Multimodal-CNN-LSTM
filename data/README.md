# 📂 ADNI Dataset Directory

## ⚠️ IMPORTANT: Data Not Included

The ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset is **NOT included** in this repository due to strict privacy and data use agreements.

---

## 🔐 How to Obtain ADNI Data

### 1. Request Access
1. Visit [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)
2. Click **"Apply for Access"**
3. Create an account and complete the application
4. Sign the **Data Use Agreement**
5. Wait for approval (typically 1-2 weeks)

### 2. Required Files
Once approved, download the following files from ADNI:

```
data/
├── ADNI1_Annual 2 Yr 3T.zip          # MRI scans (3-Tesla T1-weighted)
├── ADNI1_Annual 2 Yr 3T1.zip         # Additional MRI scans
├── MMSE_20Nov2025.csv                # Mini-Mental State Examination scores
├── DXSUM_20Nov2025.csv               # Diagnostic summary
└── ADNI1_Annual_2_Yr_3T_11_20_2025.csv  # MRI metadata
```

### 3. Extract Data
After downloading, place the zip files in this `data/` folder and run:

```bash
python scripts/extract_zip.py
```

Or manually extract:
```bash
unzip "ADNI1_Annual 2 Yr 3T.zip" -d data/
```

---

## 📊 Expected Directory Structure

After extraction, your directory should look like:

```
data/
├── ADNI/                      # MRI scan folders
│   ├── 002_S_1018/
│   │   └── ... .nii.gz files
│   ├── 002_S_4225/
│   └── ...
├── MMSE_20Nov2025.csv
├── DXSUM_20Nov2025.csv
└── ADNI1_Annual_2_Yr_3T_11_20_2025.csv
```

---

## 🚫 What NOT to Do

- ❌ **DO NOT** share ADNI data publicly
- ❌ **DO NOT** commit data to GitHub
- ❌ **DO NOT** upload to cloud storage without encryption
- ✅ **DO** keep data on secure local systems
- ✅ **DO** comply with ADNI Data Use Agreement

---

## 🆘 Alternative: Use Demo Data

If you cannot access ADNI, consider using:

### Option 1: OASIS Brain Dataset (Publicly Available)
- Website: [https://www.oasis-brains.org/](https://www.oasis-brains.org/)
- Free registration required
- Similar structure to ADNI

### Option 2: Synthetic Demo Data
Generate synthetic data for testing the pipeline:

```python
# Create dummy MRI volumes
import numpy as np
import nibabel as nib

dummy_mri = np.random.randn(64, 64, 64).astype(np.float32)
nifti_img = nib.Nifti1Image(dummy_mri, affine=np.eye(4))
nib.save(nifti_img, 'data/demo_subject_001.nii.gz')
```

---

## 📧 Need Help?

If you have trouble accessing ADNI data:
1. Contact ADNI support: [adni-info@loni.usc.edu](mailto:adni-info@loni.usc.edu)
2. Check ADNI documentation: [https://adni.loni.usc.edu/data-samples/](https://adni.loni.usc.edu/data-samples/)
3. Open an issue in this repository for code-related questions

---

## 📜 Data Use Agreement

By using ADNI data, you agree to:
- Use data solely for research purposes
- Not redistribute data to third parties
- Cite ADNI in all publications
- Follow ethical research guidelines

**Citation:**
```
Data used in preparation of this article were obtained from the 
Alzheimer's Disease Neuroimaging Initiative (ADNI) database 
(adni.loni.usc.edu). As such, the investigators within the ADNI 
contributed to the design and implementation of ADNI and/or provided 
data but did not participate in analysis or writing of this report.
```