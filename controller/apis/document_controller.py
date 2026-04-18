from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from service.pdf.document_indexer import get_document_indexing_service

router = APIRouter()


class DocumentIndexRequest(BaseModel):
    doc_source: str = ''
    pdf_path: str
    force_rebuild: bool = False
    collection_name: str = "default"


@router.post("/documents/index")
async def index_documents(request: DocumentIndexRequest):
    return await get_document_indexing_service().index_documents(
        pdf_path=request.pdf_path,
        force_rebuild=request.force_rebuild,
        collection_name=request.collection_name,
        doc_source=request.doc_source or None,
    )

