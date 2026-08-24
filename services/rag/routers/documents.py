from io import BytesIO

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from rag.container import get_rag_service
from rag.schemas.documents import (
    DocumentResponse,
    DocumentUpload,
    QueryRequest,
    QueryResponse,
)
from rag.services.rag_service import RAGService
from shared.utils.response import paginated_response

router = APIRouter(prefix="/api")


def _extract_text(file_bytes: bytes, filename: str = "", content_type: str = "") -> str:
    is_pdf = (
        file_bytes[:5] == b"%PDF-"
        or filename.lower().endswith(".pdf")
        or "pdf" in (content_type or "").lower()
    )
    if is_pdf:
        try:
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            extracted = "\n\n".join(pages).strip()
            if extracted:
                return extracted
        except Exception:
            pass
        # The file is a PDF but no text could be extracted (encrypted,
        # malformed, or a scanned/image-only PDF). Surface this instead of
        # silently ingesting binary garbage decoded as UTF-8.
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not extract text from the PDF. It may be encrypted, "
                "corrupted, or a scanned/image-only document."
            ),
        )
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("utf-8", errors="replace")


@router.post("/documents/upload", response_model=DocumentResponse, status_code=201)
async def upload_document_file(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    source: str = Form(default=""),
    service: RAGService = Depends(get_rag_service),
):
    raw = await file.read()
    content = _extract_text(raw, file.filename or "", file.content_type or "")
    doc_title = title or file.filename or "Untitled"
    data = DocumentUpload(
        title=doc_title,
        content=content,
        source=source or file.filename,
        content_type=file.content_type or "text/plain",
    )
    return await service.ingest_document(data)


@router.post("/documents", response_model=DocumentResponse, status_code=201)
async def upload_document_json(
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
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
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
