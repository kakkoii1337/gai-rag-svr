import os
import uuid
from gai.lib.common.errors import DuplicatedDocumentException
from gai.rag.server.dalc.IndexedDocumentChunk import IndexedDocumentChunk
from gai.rag.server.dalc.IndexedDocumentChunkGroup import IndexedDocumentChunkGroup
from gai.rag.server.dtos.chunk_info import ChunkInfoPydantic
from gai.rag.server.dtos.indexed_doc_chunkgroup import IndexedDocChunkGroupPydantic
from gai.rag.server.dtos.indexed_doc_chunk import IndexedDocChunkPydantic
from gai.rag.server.dtos.indexed_doc_header import IndexedDocHeaderPydantic
from gai.rag.server.dtos.indexed_doc import IndexedDocPydantic

from tqdm import tqdm
from datetime import datetime
from datetime import date
from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import sessionmaker, selectinload, defer,joinedload, scoped_session
from gai.rag.server.dalc.Base import Base
from gai.lib.common.utils import get_gai_config, get_app_path
from gai.lib.common import logging, file_utils
from gai.lib.common.PDFConvert import PDFConvert
from gai.rag.server.dalc.IndexedDocument import IndexedDocument
logger = logging.getLogger(__name__)
from sqlalchemy.orm import Session

from contextlib import contextmanager
@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()    

class RAGDBRepository:
    
    def __init__(self, engine=None, session_factory=None):
        self.config = get_gai_config()["gen"]["rag-instructor-sentencepiece"]
        self.app_path = get_app_path()
        self.chunks_path = os.path.join(self.app_path, self.config["chunks"]["path"])
        self.chunk_size = self.config["chunks"]["size"]
        self.chunk_overlap = self.config["chunks"]["overlap"]

        # These states are used by the context manager only. Do not use directly.
        if session_factory is not None:
            self.session_factory = session_factory
        elif engine is not None:
            self.session_factory = scoped_session(sessionmaker(bind=engine))
        else:
            raise ValueError("Either an engine or session_factory must be provided")        

    @contextmanager
    def session_scope(self, external_session=None):
        """Provide a transactional scope around a series of operations."""
        session = external_session if external_session else self.session_factory()
        own_session = external_session is None
        try:
            yield session
            if own_session:
                session.commit()
        except Exception as e:
            if own_session:
                session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            if own_session:
                session.close()

# Indexing Transaction -------------------------------------------------------------------------------------------------------------------------------------------

    '''
    This function will either load a PDF file or text file into memory.
    '''
    def _load_and_convert(self, file_path, file_type=None):
        if file_type is None:
            file_type = file_path.split('.')[-1]
        if file_type == 'pdf':
            text = PDFConvert.pdf_to_text(file_path)
        else:
            with open(file_path, 'r') as f:
                text = f.read()
        return text

    '''
    Used to get document_id from content
    '''
    def create_document_hash(self, file_path):
        text = self._load_and_convert(file_path)
        return file_utils.create_chunk_id_base64(text)


# Collections -------------------------------------------------------------------------------------------------------------------------------------------
    # Collection is just a grouping of documents and is not a record in RAGDBRepository.
    # However, Collection is a record in the RAGVSRepository.

    def purge(self):
        metadata = MetaData()
        metadata.reflect(bind=self.engine)
        metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(self.engine)

    '''
    Delete all the documents (and its chunks) with the given collection name.
    '''
    def delete_collection(self, collection_name, session=None):
        with self.session_scope(session) as session:        
            try:
                documents = session.query(IndexedDocument).filter_by(CollectionName=collection_name).all()
                for document in documents:
                    self.delete_document(collection_name=collection_name, document_id=document.Id)
            except Exception as e:
                logger.error(f"RAGDBRepository: Error deleting collection {collection_name}. Error={str(e)}")
                raise

    def collection_chunk_count(self,collection_name):
        with self.session_scope() as session:        
            try:
                chunks = session.query(IndexedDocumentChunk).join(IndexedDocumentChunkGroup).join(IndexedDocument).filter(IndexedDocument.CollectionName==collection_name).all()
                return len(chunks)
            except Exception as e:
                logger.error(f"RAGDBRepository: Error getting chunk count for collection {collection_name}. Error={str(e)}")
                raise

# Documents -------------------------------------------------------------------------------------------------------------------------------------------

    def list_document_headers(self, collection_name=None) -> list[IndexedDocPydantic]:
        with self.session_scope() as session:        
            try:
                if collection_name is None:
                    documents = session.query(IndexedDocument).all()
                else:
                    documents = session.query(IndexedDocument).filter_by(CollectionName=collection_name).all()

                return [document.to_pydantic() for document in documents]
            except Exception as e:
                logger.error(f"RAGDBRepository.list_document_headers: Error = {e}")
                raise



# Document -------------------------------------------------------------------------------------------------------------------------------------------

    '''
    This will read file and create a document header and generate the document ID.
    The document header is the original source and is decoupled from the chunks.
    This is because there will be many different ways of splitting the document into chunks but the source will remain the same.
    It is also possible that the document may be deactivated from the semantic search and when that is the case, all chunk groups
    (as well as the chunks under each group) will be deactivated as well.
    '''
    def create_document_header(self,
        collection_name,
        file_path,
        file_type,
        title=None,
        source=None,
        abstract=None,
        authors=None,
        publisher=None,
        published_date=None,
        comments=None,
        keywords=None,
        session=None
        ):
        with self.session_scope(session) as session:
            try:
                document = IndexedDocument()

                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                document_id = self.create_document_hash(file_path)
                existing_doc = session.query(IndexedDocument).filter(
                    IndexedDocument.Id==document_id, 
                    IndexedDocument.CollectionName==collection_name
                    ).first()
                if existing_doc is not None:
                    raise DuplicatedDocumentException(f"Document already exists in the database. document_id: {document_id}")
                
                document.Id = document_id
                document.FileName = os.path.basename(file_path)
                document.FileType = file_type
                document.ByteSize = os.path.getsize(file_path)
                document.CollectionName = collection_name
                document.Title = title
                document.Source = source
                document.Abstract = abstract
                document.Authors = authors
                document.Publisher = publisher
                document.PublishedDate = published_date
                document.Comments = comments
                document.Keywords = keywords
                document.IsActive = True
                document.CreatedAt = datetime.now()
                document.UpdatedAt = datetime.now()

                # Assuming document.PublishedDate could be either a string or a datetime.date object
                if isinstance(document.PublishedDate, str):
                    try:
                        # Correct the date format to match the input, e.g., '2017-June-12'
                        document.PublishedDate = datetime.strptime(document.PublishedDate, '%Y-%B-%d').date()
                    except ValueError:
                        # Log the error or handle it as needed
                        document.PublishedDate = None
                elif not isinstance(document.PublishedDate, date):
                    document.PublishedDate = None

                # Read the file content
                with open(file_path, 'rb') as f:
                    document.File = f.read()
                
                session.add(document)
                session.flush()
                return document.Id                

            except Exception as e:
                logger.error(f"RAGDBRepository.create_document_header: Error={str(e)}")
                raise


    def get_document_header(self, collection_name, document_id)->IndexedDocPydantic:
        with self.session_scope() as session:
            try:
                orm = session.query(IndexedDocument).options(
                    selectinload(IndexedDocument.ChunkGroups),
                    defer(IndexedDocument.File)
                ).filter(
                    IndexedDocument.Id==document_id, 
                    IndexedDocument.CollectionName==collection_name
                ).first()
                if orm is None:
                    return None
                pydantic=orm.to_pydantic()

                #exclude the file content
                pydantic.File = None
                return pydantic
            except Exception as e:
                logger.error(f"RAGDBRepository.get_document_header: Error = {e}")
                raise

    def update_document_header(self, 
                collection_name, 
                document_id, 
                title=None, 
                source=None, 
                abstract=None,
                authors=None,
                publisher=None,
                published_date=None, 
                comments=None,
                keywords=None,
                session=None):
        with self.session_scope(session) as session:
            try:
                existing_doc = session.query(IndexedDocument).filter(
                    IndexedDocument.Id==document_id, 
                    IndexedDocument.CollectionName==collection_name
                ).first()

                if existing_doc is not None:
                    if title is not None:
                        existing_doc.Title = title
                    if source is not None:
                        existing_doc.Source = source
                    if abstract is not None:
                        existing_doc.Abstract = abstract
                    if authors is not None:
                        existing_doc.Authors = authors
                    if publisher is not None:
                        existing_doc.Publisher = publisher
                    if published_date is not None:
                        if published_date and isinstance(published_date, str):
                            try:
                                existing_doc.PublishedDate = datetime.strptime(published_date, '%Y-%b-%d')
                            except:
                                existing_doc.PublishedDate = None
                        else:
                            existing_doc.PublishedDate = None
                    if comments is not None:
                        existing_doc.Comments = comments
                    if keywords is not None:
                        existing_doc.Keywords = keywords
                    existing_doc.UpdatedAt = datetime.now()
                    logger.info(f"RAGDBRepository.update_document_header: document updated successfully. DocumentId={document_id}")
                    return document_id
                else:
                    raise ValueError("No document found with the provided Id.")
            except Exception as e:
                logger.error(f"RAGDBRepository.update_document_header: Error = {e}")
                raise

    def delete_document(self, collection_name, document_id, session=None):
        with self.session_scope(session) as session:
            try:
                document = session.query(IndexedDocument).filter(
                    IndexedDocument.Id==document_id, 
                    IndexedDocument.CollectionName==collection_name
                ).first()
                if document is None:
                    raise ValueError("No document found with the provided Id.")
                for chunk_group in document.ChunkGroups:
                    for chunk in chunk_group.Chunks:
                        session.delete(chunk)
                    session.delete(chunk_group)
                session.delete(document)
            except Exception as e:
                logger.error(f"RAGDBRepository.delete_document: Error = {e}")
                raise

    # This method retrieves the file content of a specific document
    def get_document_file(self, collection_name, document_id):
        with self.session_scope() as session:
            try:
                orm = session.query(IndexedDocument).options(
                        joinedload(IndexedDocument.ChunkGroups).lazyload('*'),  # Corrected to use proper relationship loading strategy
                        defer(IndexedDocument.Abstract),     # Optionally defer other large fields not required
                        defer(IndexedDocument.Comments),
                        defer(IndexedDocument.Keywords)
                    ).filter(
                        IndexedDocument.Id == document_id,
                        IndexedDocument.CollectionName == collection_name
                    ).first()
                if orm is None:
                    return None
                return orm.File  # Returns the binary content of the file
            except Exception as e:
                logger.error(f"RAGDBRepository.get_document_file: Error = {e}")
                raise


# ChunkGroups -------------------------------------------------------------------------------------------------------------------------------------------

    '''
    There are many ways that a document can be chunked based on different strategies such as chunk size, overlap, algorithm, etc.
    This function will create a chunk group and then create the chunks in chunks_dir based on the strategy.
    '''
    def create_chunkgroup(self, 
                          collection_name, 
                          document_id, 
                          chunk_size, 
                          chunk_overlap, 
                          splitter,
                          session=None) -> IndexedDocChunkGroupPydantic:
        with self.session_scope(session) as session:            
            existing_doc = session.query(IndexedDocument).filter(
                IndexedDocument.Id==document_id, 
                IndexedDocument.CollectionName==collection_name
            ).first()

            if existing_doc is None:
                raise ValueError(f"RAGDBRepository.create_chunkgroup: Document header not found {document_id}")
            
            if existing_doc.FileType.lower() == 'pdf' or existing_doc.FileType.lower() == '.pdf':
                import tempfile
                with tempfile.NamedTemporaryFile() as temp_file:
                    temp_file.write(existing_doc.File)
                    temp_file_path = temp_file.name
                    text = PDFConvert.pdf_to_text(temp_file_path)
            else:
                text = existing_doc.File.decode('utf-8')
            
            filename = ".".join(existing_doc.FileName.split(".")[:-1])
            src_file = f"/tmp/{filename}.txt"
            with open(src_file, 'w') as f:
                f.write(text)

            # TODO: This reads an entire file into memory and is not efficient for handling large files.
            # It would be better to pass a stream to the splitter instead.
            chunks_dir = splitter(text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks_count = len(os.listdir(chunks_dir))

            chunkgroup = IndexedDocumentChunkGroup()
            chunkgroup.Id = str(uuid.uuid4())
            chunkgroup.DocumentId = document_id
            chunkgroup.SplitAlgo = "recursive_split"
            chunkgroup.ChunkCount = chunks_count
            chunkgroup.ChunkSize = chunk_size
            chunkgroup.Overlap = chunk_overlap
            chunkgroup.IsActive = True
            chunkgroup.ChunksDir = chunks_dir

            session.add(chunkgroup)
            pydantic_chunkgroup = chunkgroup.to_pydantic()
            return pydantic_chunkgroup

    def list_chunkgroup_ids(self, document_id=None):
        with self.session_scope() as session:
            if document_id is not None:
                return session.query(IndexedDocumentChunkGroup.Id).filter_by(DocumentId=document_id).all()
            return session.query(IndexedDocumentChunkGroup.Id).all()

    def get_chunkgroup(self, chunkgroup_id):
        with self.session_scope() as session:        
            chunkgroup= session.query(IndexedDocumentChunkGroup).filter_by(Id=chunkgroup_id).first()
            if chunkgroup:
                session.expunge(chunkgroup)
            return chunkgroup

    def list_chunkgroups_by_chunkhash(self, chunk_hash):
        with self.session_scope() as session:        
            return session.query(IndexedDocumentChunkGroup).join(IndexedDocumentChunk).filter(IndexedDocumentChunk.ChunkHash==chunk_hash).all()

    def delete_chunkgroup(self, chunk_group_id, session=None):
        with self.session_scope(session) as session:        
            chunkgroup = session.query(IndexedDocumentChunkGroup).filter_by(Id=chunk_group_id).first()
            if chunkgroup is None:
                raise ValueError("No chunk group found with the provided Id.")
            for chunk in chunkgroup.Chunks:
                session.delete(chunk)
            session.delete(chunkgroup)

# Chunks -------------------------------------------------------------------------------------------------------------------------------------------

    '''
    For each file in the chunks_dir, create the corresponding chunk in the database and add it to the chunk group.
    Returns an array of chunk info.
    '''
    def create_chunks(self, chunk_group_id, chunks_dir, session=None):
        result = []
        with self.session_scope(session) as session:        
            chunk_group = session.query(IndexedDocumentChunkGroup).filter_by(Id=chunk_group_id).first()

            chunk_ids = os.listdir(chunks_dir)
            for chunk_id in tqdm(chunk_ids):
                chunk = IndexedDocumentChunk()
                chunk.Id = str(uuid.uuid4())
                chunk.ChunkGroupId = chunk_group.Id

                # Load content
                chunk.Content = None
                with open(os.path.join(chunks_dir, chunk_id), 'rb') as f:
                    chunk.Content = f.read().decode('utf-8')
                    chunk.ByteSize = len(chunk.Content)

                # Create chunk hash and check for mismatch
                chunk.ChunkHash = file_utils.create_chunk_id_base64(chunk.Content)
                if (chunk.ChunkHash != chunk_id):
                    raise ValueError(f"RAGDBRepository.create_chunks: Chunk hash mismatch: {chunk.ChunkHash} != {chunk_id}")

                # Check for chunk hash duplicates in DB
                found = session.query(IndexedDocumentChunk).filter_by(ChunkHash=chunk.Id).first()

                chunk.IsDuplicate = (found is not None)
                chunk.IsIndexed = False
                chunk_group.Chunks.append(chunk)
                session.add(chunk)

                result.append(ChunkInfoPydantic(
                    Id=chunk.Id,
                    ChunkHash=chunk.ChunkHash,
                    IsDuplicate=chunk.IsDuplicate,
                    IsIndexed=chunk.IsIndexed))
            return result

    def list_chunks(self, chunkgroup_id=None):
        with self.session_scope() as session:
            if chunkgroup_id is None:
                chunks = session.query(IndexedDocumentChunk).all()
            else:
                chunks = session.query(IndexedDocumentChunk).filter_by(ChunkGroupId=chunkgroup_id).all()

            # Convert each ORM object to Pydantic model using from_dalc
            return [chunk.to_pydantic() for chunk in chunks]

    def get_chunk(self, chunk_id, session=None):
        with self.session_scope(session) as session:        
            return session.query(IndexedDocumentChunk).filter_by(Id=chunk_id).first()

    def list_chunks_by_document_id(self, document_id):
            with self.session_scope() as session:
                try:
                    chunks = session.query(IndexedDocumentChunk).join(
                        IndexedDocumentChunkGroup,
                        IndexedDocumentChunk.ChunkGroupId == IndexedDocumentChunkGroup.Id
                    ).join(
                        IndexedDocument,
                        IndexedDocumentChunkGroup.DocumentId == IndexedDocument.Id
                    ).filter(
                        IndexedDocument.Id == document_id
                    ).all()
                    return [chunk.to_pydantic() for chunk in chunks]
                except Exception as e:
                    logger.error(f"RAGDBRepository.list_chunks_by_document_id: Error = {e}")
                    raise