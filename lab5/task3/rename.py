import os


def rename_files_in_folder(folder_path, name_template):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for idx, filename in enumerate(files, start=1):
        _, ext = os.path.splitext(filename)
        new_name = f"{name_template}_{idx}{ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
    print(f"Переименование завершено. Всего файлов: {len(files)}")


rename_files_in_folder(r'D:\pen_pencil_marker\marker', 'a')
rename_files_in_folder(r'D:\pen_pencil_marker\marker', 'marker')