import asyncio
import json
import os
import tempfile

os.environ["LOG_LEVEL"]="DEBUG"
from gai.lib.common.logging import getLogger
logger = getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

# GAI
from gai.lib.common.errors import *
from gai.lib.server.api_dependencies import get_app_version
from gai.lib.common.WSManager import ws_manager
from gai.rag.server.dtos.indexed_doc import IndexedDocPydantic
from gai.rag.server.dtos.create_doc_header_request import CreateDocHeaderRequestPydantic
from gai.rag.server.dtos.indexed_doc_chunkgroup import IndexedDocChunkGroupPydantic
from gai.rag.server.dtos.indexed_doc_chunk_ids import IndexedDocChunkIdsPydantic

# Router
from pydantic import BaseModel
from typing import List, Optional
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse
from fastapi import APIRouter, Body, Depends, File, Form, UploadFile, WebSocket, WebSocketDisconnect, websockets
import uuid

router = APIRouter()
pyproject_toml = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "..", "..", "..", "..", "pyproject.toml")

# Add this at the beginning, before your other routes
@router.get("/")
async def root():
    return JSONResponse(status_code=200, content={"message": "gai-rag-svr"})

### ----------------- RAG ----------------- ###

# VERSION
@router.get("/gen/v1/rag/version")
async def version():
    return JSONResponse(status_code=200, content={
        "version": get_app_version(pyproject_toml=pyproject_toml)
    })

# MULTI-STEP INDEXING -------------------------------------------------------------------------------------------------------------------------------------------

# Description: Step 1 of 3 - Save file to database and create header
# POST /gen/v1/rag/step/header
@router.post("/gen/v1/rag/step/header", response_model=IndexedDocPydantic)
async def step_header_async(file: UploadFile = File(...), req:str = Form(...)) -> IndexedDocPydantic:
    try:
        req=CreateDocHeaderRequestPydantic(**json.loads(req))
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.step_header_async: {id} Error=Failed to parse request,{str(e)}")
        raise InternalException(id)

    rag = app.state.host.generator
    logger.info(f"rag_api.index_file: started.")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            # Save the file to a temporary directory using the filename from the header
            dest_filename = os.path.basename(req.FilePath)
            dest_filepath = os.path.join(temp_dir, dest_filename)
            with open(dest_filepath, "wb+") as file_object:
                content = await file.read()  # Read the content of the uploaded file
                file_object.write(content)
                # Update the request with the new file path
                req.FilePath = dest_filepath
            logger.info(f"rag_api.step_header_async: temp file created.")

            # Give the temp file path to the RAG
            #metadata_dict = json.loads(metadata)

            doc = await rag.index_document_header_async(req)
            return doc
    except DuplicatedDocumentException:
        raise
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.step_header_async: {id} Error=Failed to create document header,{str(e)}")
        raise InternalException(id)


# Description: Step 2 of 3 - Split file and save chunks to database
# POST /gen/v1/rag/step/split
class DocumentSplitRequest(BaseModel):
    collection_name:str
    document_id: str
    chunk_size: int
    chunk_overlap: int
@router.post("/gen/v1/rag/step/split", response_model=IndexedDocChunkGroupPydantic)
async def step_split_async(req: DocumentSplitRequest) -> IndexedDocChunkGroupPydantic:
    logger.info(f"rag_api.index_file: started.")
    rag = app.state.host.generator
    try:
        chunkgroup = await rag.index_document_split_async(
            collection_name=req.collection_name,
            document_id=req.document_id, 
            chunk_size=req.chunk_size, 
            chunk_overlap=req.chunk_overlap)
        return chunkgroup
    except DuplicatedDocumentException:
        raise
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.step_split_async: {id} Error=Failed to split document,{str(e)}")
        raise InternalException(id)

# Description: Step 3 of 3 - Index chunks into vector database
# POST /gen/v1/rag/step/index
class DocumentIndexRequest(BaseModel):
    collection_name:str
    document_id: str
    chunkgroup_id: str
@router.post("/gen/v1/rag/step/index")
async def step_index_async(req: DocumentIndexRequest) -> IndexedDocChunkIdsPydantic:
    logger.info(f"rag_api.index_file: started.")
    rag = app.state.host.generator
    try:
        result = await rag.index_document_index_async(
            collection_name=req.collection_name,
            document_id=req.document_id, 
            chunkgroup_id = req.chunkgroup_id,
            ws_manager=ws_manager
            )
        return result
    except DuplicatedDocumentException:
        raise
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.step_index_async: {id} Error=Failed to index chunks,{str(e)}")
        raise InternalException(id)

# SINGLE-STEP INDEXING -------------------------------------------------------------------------------------------------------------------------------------------

# Description : This indexes the entire file in a single step.
# POST /gen/v1/rag/index-file
@router.post("/gen/v1/rag/index-file")
async def index_file_async(file: UploadFile = File(...), req: str = Form(...))  -> IndexedDocChunkIdsPydantic:
    logger.info(f"rag_api.index_file: started.")

    try:
        req=CreateDocHeaderRequestPydantic(**json.loads(req))
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.index_file_async: {id} Error=Failed to parse request,{str(e)}")
        raise InternalException(id)

    rag = app.state.host.generator
    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            # Save the file to a temporary directory using the filename from the header
            dest_filename = os.path.basename(req.FilePath)
            dest_filepath = os.path.join(temp_dir, dest_filename)

            with open(dest_filepath, "wb+") as file_object:
                content = await file.read()  # Read the content of the uploaded file
                file_object.write(content)
                # Update the request with the new file path
                req.FilePath = dest_filepath
            logger.info(f"rag_api.index_file_async: temp file created.")
            result = await rag.index_async(req, ws_manager)

            return result
    except DuplicatedDocumentException:
        raise
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.index_file_async: {id} {str(e)}")
        raise InternalException(id)


# WEBSOCKET -------------------------------------------------------------------------------------------------------------------------------------------

# WEBSOCKET "/gen/v1/rag/index-file/ws/{agent_id}"
# This endpoint is only required for maintaining a websocket connection to provide real-time status updates.
# The actual work is done by ws_manager.
@router.websocket("/gen/v1/rag/index-file/ws/{agent_id}")
async def index_file_websocket_async(websocket:WebSocket,agent_id=None):
    await ws_manager.connect(websocket,agent_id)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        await ws_manager.disconnect(agent_id)
    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"websocket closed.")

# RETRIEVAL -------------------------------------------------------------------------------------------------------------------------------------------

# Retrieve document chunks using semantic search
# POST /gen/v1/rag/retrieve
class QueryRequest(BaseModel):
    collection_name: str
    query_texts: str
    n_results: int = 3
@router.post("/gen/v1/rag/retrieve")
async def retrieve(request: QueryRequest = Body(...)):
    rag = app.state.host.generator
    try:
        logger.info(
            f"main.retrieve: collection_name={request.collection_name}")
        result = rag.retrieve(collection_name=request.collection_name,
                              query_texts=request.query_texts, n_results=request.n_results)
        logger.debug(f"main.retrieve={result}")
        return JSONResponse(status_code=200, content={
            "retrieved": result
        })
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.retrieve: {id} {str(e)}")
        raise InternalException(id)


#Collections-------------------------------------------------------------------------------------------------------------------------------------------

# GET /gen/v1/rag/collections
@router.get("/gen/v1/rag/collections")
async def list_collections():
    rag = app.state.host.generator
    try:
        collections = [collection.name for collection in rag.list_collections()]
        return JSONResponse(status_code=200, content={
            "collections": collections
        })
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.list_collections: {id} {str(e)}")
        raise InternalException(id)

# DELETE /gen/v1/rag/purge
@router.delete("/gen/v1/rag/purge")
async def purge_all():
    rag = app.state.host.generator
    try:
        rag.purge_all()
        return JSONResponse(status_code=200, content={
            "message": "Purge successful."
        })
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.purge: {id} {str(e)}")
        raise InternalException(id)

# DELETE /gen/v1/rag/collection/{}
# Response:
# - 200: { "count": remaining collections }
# - 404: { "message": "Collection with name={collection_name} not found" }
# - 500: { "message": "Internal error: {id}" }
@router.delete("/gen/v1/rag/collection/{collection_name}")
async def delete_collection(collection_name):
    rag = app.state.host.generator
    try:
        if collection_name not in [collection.name for collection in rag.list_collections()]:
            logger.warning(f"rag_api.delete_collection: Collection with name={collection_name} not found.")
            raise CollectionNotFoundException(collection_name)

        rag.delete_collection(collection_name=collection_name)
        after = rag.list_collections()
        return {
            "count": len(after)
        }
    except CollectionNotFoundException:
        raise
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.delete_collection: {id} {str(e)}")
        raise InternalException(id)


#Documents-------------------------------------------------------------------------------------------------------------------------------------------

# GET /gen/v1/rag/documents
@router.get("/gen/v1/rag/documents",response_model=list[IndexedDocPydantic])
async def list_document_headers()-> list[IndexedDocPydantic]:
    rag = app.state.host.generator
    try:
        docs = rag.list_document_headers()
        result = []
        for doc in docs:
            result.append(doc)
        return result   
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.list_documents: {id} {str(e)}")
        raise InternalException(id)

# GET /gen/v1/rag/collection/{collection_name}/documents
@router.get("/gen/v1/rag/collection/{collection_name}/documents",response_model=List[IndexedDocPydantic])
async def list_document_headers_by_collection(collection_name) -> List[IndexedDocPydantic]:
    rag = app.state.host.generator
    try:
        docs = rag.list_document_headers(collection_name=collection_name)
        result = []
        for doc in docs:
            result.append(doc)
        return result
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.list_documents: {id} {str(e)}")
        raise InternalException(id)

#Document-------------------------------------------------------------------------------------------------------------------------------------------

# GET /gen/v1/rag/collection/{collection_name}/document/{document_id}
# Description: Get only the document details and chunkgroup without the file blob. To load the file, load chunks using chunkgroup_id.
# Response:
# - 200: { "document": {...} }
# - 404: { "message": "Document with id {document_id} not found" }
# - 500: { "message": "Internal error: {id}" }
@router.get("/gen/v1/rag/collection/{collection_name}/document/{document_id}",response_model=IndexedDocPydantic)
async def get_document_header(collection_name, document_id)->IndexedDocPydantic:
    rag = app.state.host.generator
    try:
        document = rag.get_document_header(collection_name=collection_name, document_id=document_id)
        if document is None:
            logger.warning(f"rag_api.get_documents: Document with Id={document_id} not found.")
            raise DocumentNotFoundException(document_id)
        
        return document
    except DocumentNotFoundException as e:
        raise e
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.get_document: {id} {str(e)}")
        raise InternalException(id)
    
# POST /gen/v1/rag/collection/{collection_name}/document/{document_id}
# Description: Same as get_document_header but by passing in the file instead of the document_id
# Response:
# - 200: { "document": {...} }
# - 404: { "message": "Document with id {document_id} not found" }
# - 500: { "message": "Internal error: {id}" }
@router.post("/gen/v1/rag/collection/{collection_name}/document/exists",response_model=IndexedDocPydantic)
async def get_document_header_by_file(collection_name: str, file: UploadFile = File(...))->IndexedDocPydantic:
    rag = app.state.host.generator
    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            # Save the file to a temporary directory
            file_location = os.path.join(temp_dir, file.filename)
            with open(file_location, "wb+") as file_object:
                content = await file.read()  # Read the content of the uploaded file
                file_object.write(content)
            logger.info(f"rag_api.step_header_async: temp file created.")

            # get document_id
            document_id = rag.create_document_hash(file_location)

            # check if document exists
            document = rag.get_document_header(collection_name, document_id)
            if document is None:
                logger.warning(f"rag_api.get_documents: Document with Id={document_id} not found.")
                raise DocumentNotFoundException(document_id)
            
            return document

    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.step_header_async: {id} Error=Failed to check document exist,{str(e)}")
        raise InternalException(id)    

# PUT /gen/v1/rag/document
# Response:
# - 200: { "message": "Document updated successfully", "document": {...} }
# - 404: { "message": "Document with id {document_id} not found" }
# - 500: { "message": "Internal error: {id}" }
class UpdateDocumentRequest(BaseModel):
    FileName: str = None
    Source: str = None
    Abstract: str = None
    Authors: str = None
    Title: str = None
    Publisher: str = None
    PublishedDate: str = None
    Comments: str = None
    Keywords: str = None
@router.put("/gen/v1/rag/collection/{collection_name}/document/{document_id}")
async def update_document_header(collection_name, document_id, req: UpdateDocumentRequest = Body(...)):
    rag = app.state.host.generator
    try:
        doc = rag.get_document_header(collection_name=collection_name, document_id=document_id)
        if doc is None:
            logger.warning(f"Document with collection_name={collection_name} document_id={document_id} not found.")
            # Bypass the error handler and return a 404 directly
            raise DocumentNotFoundException(document_id)
                
        updated_doc = rag.update_document_header(collection_name, document_id, **req.dict())
        return JSONResponse(status_code=200, content={
            "message": "Document updated successfully",
            "document": updated_doc
        })
    except DocumentNotFoundException as e:
        raise e
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.update_document: {id} {str(e)}")
        raise InternalException(id)

# DELETE /gen/v1/rag/document/{document_id}
# Response:
# - 200: { "message": "Document with id {document_id} deleted successfully" }
# - 404: { "message": "Document with id {document_id} not found" }
# - 500: { "message": "Internal error: {id}" }
@router.delete("/gen/v1/rag/collection/{collection_name}/document/{document_id}")
async def delete_document(collection_name, document_id):
    rag = app.state.host.generator
    try:
        doc = rag.get_document_header(collection_name=collection_name, document_id=document_id)
        if doc is None:
            logger.warning(f"Document with Id={document_id} not found.")
            raise DocumentNotFoundException(document_id)
        
        rag.delete_document(collection_name=collection_name, document_id=document_id)
        return JSONResponse(status_code=200, content={
            "message": f"Document with id {document_id} deleted successfully"
        })
    except DocumentNotFoundException as e:
        raise e
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.delete_document: {id} {str(e)}")
        raise InternalException(id)

# GET /gen/v1/rag/collection/{collection_name}/document/{document_id}/file
# Description: Retrieves the file content of a specific document by document_id
# Response:
# - 200: Binary content of the file
# - 404: { "message": "Document with id {document_id} not found" }
# - 500: { "message": "Internal error: {id}" }
@router.get("/gen/v1/rag/collection/{collection_name}/document/{document_id}/file")
async def get_document_file(collection_name: str, document_id: str):
    rag = app.state.host.generator
    try:
        # Fetch the file content
        file_content = rag.get_document_file(collection_name, document_id)
        if file_content is None:
            logger.warning(f"Document with collection_name={collection_name} document_id={document_id} not found.")
            raise DocumentNotFoundException(document_id)
        
        # Save the file content to a temporary file and return it
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        return FileResponse(temp_file_path, filename=f"{document_id}.pdf")  # Adjust the filename and content type as needed
    except DocumentNotFoundException as e:
        raise e
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.get_document_file: {id} {str(e)}")
        raise InternalException(id)


### ----------------- CHUNKGROUPS ----------------- ###
# GET "/gen/v1/rag/chunkgroups"
@router.get("/gen/v1/rag/chunkgroups")
async def list_chunkgroup_ids_async():
    rag = app.state.host.generator
    try:
        chunkgroups = rag.list_chunkgroup_ids()
        result = []
        for chunkgroup in chunkgroups:
            dict = jsonable_encoder(chunkgroup)
            result.append(dict)
        return {
            "chunkgroups": result
        }
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.list_chunkgroups: {id} {str(e)}")
        raise InternalException(id)

# GET /gen/v1/rag/chunkgroup/{chunkgroup_id}
@router.get("/gen/v1/rag/chunkgroup/{chunkgroup_id}")
async def get_chunkgroup_async(chunkgroup_id):
    rag = app.state.host.generator
    try:
        chunkgroup = rag.get_chunkgroup(chunkgroup_id)
        if chunkgroup is None:
            logger.warning(f"rag_api.get_chunkgroup: Chunkgroup with Id={chunkgroup_id} not found.")
            raise ChunkgroupNotFoundException(chunkgroup_id)
        
        result = jsonable_encoder(chunkgroup)
        return {
            "chunkgroup": result
        }
    except ChunkgroupNotFoundException as e:
        raise e
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.get_chunkgroup: {id} {str(e)}")
        raise InternalException(id)

# DELETE "/gen/v1/rag/collection/{collection_name}/chunkgroup/{chunkgroup_id}"
@router.delete("/gen/v1/rag/collection/{collection_name}/chunkgroup/{chunkgroup_id}")
async def delete_chunkgroup_async(collection_name, chunkgroup_id):
    rag = app.state.host.generator
    try:
        rag.delete_chunkgroup(collection_name, chunkgroup_id)
        return {
            "message": f"Chunkgroup with id {chunkgroup_id} deleted successfully"
        }
    except ChunkgroupNotFoundException as e:
        raise e
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.delete_chunkgroup: {id} {str(e)}")
        raise InternalException(id)

### ----------------- CHUNKS ----------------- ###

# GET /gen/v1/rag/chunks/{chunkgroup_id}
# Descriptions: Use this to get chunk ids only from a group
@router.get("/gen/v1/rag/chunks/{chunkgroup_id}")
async def list_chunks_by_chunkgroup_id_async(chunkgroup_id: str):
    rag = app.state.host.generator
    chunks = rag.list_chunks(chunkgroup_id)
    return chunks

# GET /gen/v1/rag/chunks
# Descriptions: Use this to get all chunk ids
@router.get("/gen/v1/rag/chunks")
async def list_chunks_async():
    rag = app.state.host.generator
    # Return chunks filtered by chunkgroup_id if provided
    chunks = rag.list_chunks()
    return chunks

# GET /gen/v1/rag/collection/{collection_name}/document/{document_id}/chunks
# Descriptions: Use this to get chunks of a document from db and vs
@router.get("/gen/v1/rag/collection/{collection_name}/document/{document_id}/chunks")
async def list_document_chunks_async(collection_name, document_id: str):
    rag = app.state.host.generator
    chunks = rag.list_document_chunks(collection_name, document_id)
    return chunks

# GET "/gen/v1/rag/collection/{collection_name}/chunk/{chunk_id}"
# Use this to get a chunk from db and vs
@router.get("/gen/v1/rag/collection/{collection_name}/chunk/{chunk_id}")
async def get_document_chunk_async(collection_name,chunk_id):
    rag = app.state.host.generator
    chunk = rag.get_document_chunk(collection_name,chunk_id)
    return chunk

# -----------------------------------------------------------------------------------------------------------------

# __main__
if __name__ == "__main__":

    # # Run self-test before anything else
    # import os
    # if os.environ.get("SELF_TEST",None):
    #     self_test_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),"self-test.py")
    #     import subprocess,sys
    #     try:
    #         subprocess.run([f"python {self_test_file}"],shell=True,check=True)
    #     except subprocess.CalledProcessError as e:
    #         sys.exit(1)
    #     ## passed self-test

    import uvicorn
    from gai.lib.server import api_factory
    from gai.lib.common import utils

    # Check if a local gai.yml exists. If not, use the default one in ~/.gai
    here = os.path.dirname(__file__)
    local_config_path = os.path.join(here, "gai.yml")
    gai_config = utils.get_gai_config()
    if os.path.exists(local_config_path):
        gai_config = utils.get_gai_config(local_config_path)    

    app = api_factory.create_app(pyproject_toml, category="rag",gai_config=gai_config)
    app.include_router(router, dependencies=[Depends(lambda: app.state.host)])
    config = uvicorn.Config(
        app=app, 
        host="0.0.0.0", 
        port=12036, 
        timeout_keep_alive=180,
        timeout_notify=150,
        workers=1
    )
    server = uvicorn.Server(config=config)
    server.run()
