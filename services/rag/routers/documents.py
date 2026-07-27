from fastapi import APIRouter, Depends, UploadFile, File

from rag.container import get_rag_service
from rag.schemas.documents import (
    DocumentUpload,
    DocumentResponse,
    QueryRequest,
    QueryResponse,
)
from rag.services.rag_service import RAGService
from shared.utils.response import paginated_response

router = APIRouter(prefix="/api")


@router.post("/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(
    data: DocumentUpload,
    service: RAGService = Depends(get_rag_service),
):
    return await service.ingest_document(data)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    service: RAGService = Depends(get_rag_service),
):
    return await service.get_document(document_id)


@router.get("/documents")
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    service: RAGService = Depends(get_rag_service),
):
    docs, total = await service.list_documents(page, page_size)
    return paginated_response(
        items=[d.model_dump(mode="json") for d in docs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    service: RAGService = Depends(get_rag_service),
):
    await service.delete_document(document_id)


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service),
):
    return await service.query(request)
