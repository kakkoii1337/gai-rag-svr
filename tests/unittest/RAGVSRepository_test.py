import os, sys
from gai.lib.common import file_utils
sys.path.insert(0,os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from chromadb.utils.embedding_functions import InstructorEmbeddingFunction

from sqlalchemy import create_engine
from gai.rag.server.dalc.Base import Base

from gai.rag.server.dalc.RAGDBRepository import RAGDBRepository
from gai.rag.server.dalc.RAGVSRepository import RAGVSRepository
from gai.rag.server.dalc.RAGDBRepository import RAGDBRepository
from gai.lib.common.logging import getLogger
logger = getLogger(__name__)

from gai.lib.common.utils import get_gai_config, get_app_path

import pytest

@pytest.fixture(scope='module')
def vs_repo():
    config = get_gai_config()
    model_path = os.path.join(get_app_path(), config["gen"]["rag-instructor-sentencepiece"]["model_path"])
    device=config["gen"]["rag-instructor-sentencepiece"]["device"]
    ef = InstructorEmbeddingFunction(model_path,device)
    return RAGVSRepository.New(in_memory=True,ef=ef)

@pytest.fixture(scope='module')
def engine():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()

@pytest.fixture(scope='module')
def db_repo(engine):
    return RAGDBRepository(engine)


#Collections-------------------------------------------------------------------------------------------------------------------------------------------

def test_ut0021_manage_collections(vs_repo):
    #self.vs_repo = RAGVSRepository.New(in_memory=True)

    col = vs_repo.get_or_create_collection('t10')
    assert col.name== 't10'

    col = vs_repo.get_or_create_collection('t20')
    assert col.name== 't20'

    cols = vs_repo.list_collections()
    assert len(cols)==2

    vs_repo.delete_collection('t10')
    cols = vs_repo.list_collections()
    assert len(cols)==1

#Index and Retrieval-------------------------------------------------------------------------------------------------------------------------------------------

# def test_ut0022_index_and_retrieve(vs_repo, db_repo):

#     # Arrange
#     collection_name='demo'
#     file_path=os.path.join(os.path.dirname(__file__), "attention-is-all-you-need.pdf")
#     session=db_repo.session_factory()
#     document_id=""
#     try:
#         document_id=db_repo.create_document_header(
#             collection_name=collection_name, 
#             file_path=file_path,
#             file_type='pdf',
#             session=session
#             )
#         chunkgroup = db_repo.create_chunkgroup(
#             collection_name=collection_name,
#             document_id=document_id, 
#             chunk_size=1000, 
#             chunk_overlap=100, 
#             splitter=file_utils.split_text,
#             session=session
#             )
#         db_chunks = db_repo.create_chunks(
#             chunkgroup.Id,
#             chunkgroup.ChunksDir,
#             session=session
#         )
#         session.commit()
#     except Exception as e:
#         session.rollback()
#         raise e

#     # Act
#     doc = db_repo.get_document_header(collection_name=collection_name, document_id=document_id)

#     # For each chunk in the database for the document, index it

#     for chunk_info in db_chunks:
#         db_chunk = db_repo.get_chunk(chunk_info.Id,session=session)
#         #db_chunk=self.db_repo.get_chunk(db_chunk.Id)
#         vs_repo.index_chunk(
#             collection_name=collection_name, 
#             content=db_chunk.Content, 
#             chunk_id=db_chunk.Id, 
#             document_id=doc.Id,
#             chunkgroup_id=chunkgroup.Id,
#             source= doc.Source if doc.Source else "",
#             abstract= doc.Abstract if doc.Abstract else "",
#             title= doc.Title if doc.Title else "",
#             published_date= doc.PublishedDate if doc.PublishedDate else "",
#             keywords= doc.Keywords if doc.Keywords else ""
#         )

#         # Assert
#         vs_chunk=vs_repo.get_chunk(collection_name, db_chunk.Id)
#         assert vs_chunk is not None

#     session.close()

#     chunks = vs_repo.retrieve(collection_name, 'What is the difference between a transformer and a CNN?', n_results=3)
#     assert len(chunks) == 3

#     for i,chunk in enumerate(chunks['documents']):
#         logger.info(f"{i}: {chunk}\n")

#Chunk-------------------------------------------------------------------------------------------------------------------------------------------

def test_ut0023_compare_collection_chunk_count(vs_repo,db_repo):
    vs_count = vs_repo.collection_chunk_count('demo')

    docs = db_repo.list_document_headers('demo')
    chunk_count = 0
    for doc in docs:
        db_chunks = db_repo.list_chunks_by_document_id(doc.Id)
        chunk_count += len(db_chunks)

    assert vs_count == chunk_count

def test_ut0024_compare_document_chunk_count(vs_repo,db_repo):
    vs_count = vs_repo.document_chunk_count('demo','5a4b585a-6b0f-4302-8217-faf9d5fad391')
    db_chunks = db_repo.list_chunks_by_document_id('5a4b585a-6b0f-4302-8217-faf9d5fad391')
    assert vs_count == len(db_chunks)
    
def test_ut0025_delete_chunks_by_document(vs_repo):
    vs_repo.delete_document('demo','5a4b585a-6b0f-4302-8217-faf9d5fad391')
    vs_count = vs_repo.document_chunk_count('demo','5a4b585a-6b0f-4302-8217-faf9d5fad391')
    assert vs_count == 0

if __name__=="__main__":
    pytest.main([sys.argv[0]])
