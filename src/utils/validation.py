def validate_mri_file(file_path):
    """Validates if a file is a valid MRI image."""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.dcm')
    return file_path.lower().endswith(valid_extensions)
