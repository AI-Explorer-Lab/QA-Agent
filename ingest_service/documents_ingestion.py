import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DocumentLoader:
    """统一文档加载器，支持多种文件类型"""

    def __init__(self, directory: str, file_type: str):
        """
        初始化文档加载器

        Args:
            directory: 目录路径或文件路径
            file_type: 文件扩展名，如 'txt', 'pdf' 等
        """
        self.directory = directory
        self.file_type = file_type.lower()
        self.path = Path(directory)

    def load_documents(self) -> list:
        """
        加载文档

        Returns:
            List of document objects with metadata and content
        """
        documents = []
        files_to_process = self._get_files()

        for file_path in files_to_process:
            # PDF files: return metadata only (content will be extracted elsewhere)
            if self.file_type == 'pdf':
                documents.append({
                    'id': file_path.stem,
                    'source': str(file_path),
                    'timestamp': datetime.now().isoformat(),
                    'filename': file_path.name
                })
                continue

            # Text files: read content
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if not content.strip():
                        print(f"Warning: {file_path} is empty")
                        continue

                    documents.append({
                        'id': file_path.stem,
                        'content': content,
                        'source': str(file_path),
                        'timestamp': datetime.now().isoformat(),
                        'filename': file_path.name
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

        print(f"Loaded {len(documents)} documents")
        return documents

    def _get_files(self) -> list:
        """获取需要处理的文件列表"""
        # If it's a file, process just that file
        if self.path.is_file():
            if self.path.suffix.lower() == f'.{self.file_type}':
                return [self.path]
            else:
                print(f"File {self.path} is not a .{self.file_type} file")
                return []
        # If it's a directory, find all files of the type
        elif self.path.is_dir():
            files = list(self.path.glob(f"*.{self.file_type}"))
            if not files:
                print(f"No .{self.file_type} files found in directory {self.path}")
            return files
        else:
            print(f"Path {self.path} does not exist")
            return []


# 保留原有函数名作为向后兼容的接口
def load_zhihu_documents(directory: str) -> list:
    """Load all Zhihu answer TXT files from the specified directory."""
    loader = DocumentLoader(directory, 'txt')
    return loader.load_documents()


def load_pdf_documents(directory: str) -> list:
    """Load all PDF files from the specified directory."""
    loader = DocumentLoader(directory, 'pdf')
    return loader.load_documents()