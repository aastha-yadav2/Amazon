"""
Post-validation packaging script.
Run AFTER the test pipeline has completed and validation has passed.
1. Backs up existing baseline files.
2. Copies enhanced files to standard names.
3. Creates Silver_Strikers_submission.zip with exact required structure.
"""

import os
import sys
import shutil
import zipfile

sys.stdout.reconfigure(encoding='utf-8')

BASE = r'd:\amazon\student_resource'
OUTPUT = os.path.join(BASE, 'output')
BACKUP = os.path.join(BASE, 'backup')
SRC = r'd:\amazon\code\business_entity_resolution\src'
CODE_DIR = r'd:\amazon\code\business_entity_resolution'

def package():
    print("=" * 60)
    print("POST-VALIDATION PACKAGING")
    print("=" * 60)

    # 1. Back up existing baseline files if they exist
    os.makedirs(BACKUP, exist_ok=True)
    for fname in ['matching_results.tsv', 'candidate_pairs.tsv']:
        src_path = os.path.join(OUTPUT, fname)
        if os.path.exists(src_path):
            dst_path = os.path.join(BACKUP, fname.replace('.tsv', '_0478699.tsv'))
            shutil.copy2(src_path, dst_path)
            print(f"  Backed up: {src_path} -> {dst_path}")
        else:
            print(f"  No existing {fname} to back up.")

    # 2. Copy enhanced files to standard names
    enhanced_matching = os.path.join(OUTPUT, 'matching_results_enhanced.tsv')
    enhanced_candidate = os.path.join(OUTPUT, 'candidate_pairs_enhanced.tsv')
    std_matching = os.path.join(OUTPUT, 'matching_results.tsv')
    std_candidate = os.path.join(OUTPUT, 'candidate_pairs.tsv')

    if not os.path.exists(enhanced_matching):
        print(f"ERROR: {enhanced_matching} not found! Run the test pipeline first.")
        return
    if not os.path.exists(enhanced_candidate):
        print(f"ERROR: {enhanced_candidate} not found! Run the test pipeline first.")
        return

    shutil.copy2(enhanced_matching, std_matching)
    shutil.copy2(enhanced_candidate, std_candidate)
    print(f"  Copied enhanced matching -> {std_matching}")
    print(f"  Copied enhanced candidate -> {std_candidate}")

    # 3. Create ZIP with exact required structure
    zip_path = os.path.join(BASE, 'Silver_Strikers_submission.zip')
    print(f"\nCreating {zip_path}...")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # output/
        zf.write(std_matching, 'output/matching_results.tsv')
        zf.write(std_candidate, 'output/candidate_pairs.tsv')

        # code/business_entity_resolution/src/
        for py_file in ['blocking.py', 'features.py', 'train_model.py', 'pipeline.py']:
            src_path = os.path.join(SRC, py_file)
            if os.path.exists(src_path):
                zf.write(src_path, f'code/business_entity_resolution/src/{py_file}')
            else:
                print(f"  WARNING: {src_path} not found, skipping in ZIP.")

        # code/business_entity_resolution/README.md
        readme_path = os.path.join(CODE_DIR, 'README.md')
        if os.path.exists(readme_path):
            zf.write(readme_path, 'code/business_entity_resolution/README.md')

        # code/business_entity_resolution/requirements.txt
        req_path = os.path.join(CODE_DIR, 'requirements.txt')
        if os.path.exists(req_path):
            zf.write(req_path, 'code/business_entity_resolution/requirements.txt')

        # Documentation_template.md
        doc_path = os.path.join(BASE, 'Documentation_template.md')
        if os.path.exists(doc_path):
            zf.write(doc_path, 'Documentation_template.md')

    print(f"\nZIP created successfully: {zip_path}")
    print(f"ZIP size: {os.path.getsize(zip_path) / (1024*1024):.1f} MB")

    # List ZIP contents
    print("\nZIP contents:")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            print(f"  {info.filename} ({info.file_size:,} bytes)")

    print("\n" + "=" * 60)
    print("PACKAGING COMPLETE")
    print("=" * 60)
    print(f"  Submission ZIP : {zip_path}")
    print(f"  Backup dir     : {BACKUP}")
    print(f"  Enhanced files preserved at:")
    print(f"    {enhanced_matching}")
    print(f"    {enhanced_candidate}")


if __name__ == '__main__':
    package()
