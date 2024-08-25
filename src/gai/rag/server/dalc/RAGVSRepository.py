import os
from gai.rag.server.models.IndexedDocumentChunkPydantic import IndexedDocumentChunkPydantic
from gai.rag.server.models.IndexedDocumentPydantic import IndexedDocumentPydantic
from chromadb.config import Settings
import chromadb
from gai.lib.common.utils import get_gai_config, get_app_path
from gai.lib.common import logging
logger = logging.getLogger(__name__)
import json
from chromadb.utils.embedding_functions import InstructorEmbeddingFunction
from dotenv import load_dotenv
load_dotenv()

import pandas as pd


class RAGVSRepository:

    @staticmethod
    def New(in_memory=False,ef=None):
        try:
            app_path = get_app_path()
            config = get_gai_config()["gen"]["rag-instructor-sentencepiece"]
            chromadb_path = os.path.join(app_path, config["chromadb"]["path"])
            if in_memory:
                logger.info(f"RAGVSRepository: in_memory")
                client = chromadb.Client()
            else:
                logger.info(f"RAGVSRepository: persistent")
                client = chromadb.PersistentClient(
                    path=chromadb_path, 
                    settings=Settings(allow_reset=True))
            return RAGVSRepository(client,ef)
        
        except Exception as e:
            if not "does not exist." in str(e):
                raise e
            
    def __init__(self, client, ef):
        app_path = get_app_path()
        config = get_gai_config()["gen"]["rag-instructor-sentencepiece"]
        device = config["device"]
        self._ef = ef
        self.client = client
        self.n_results = config["chromadb"]["n_results"]

    def purge(self):
        self.client.reset()

#Collections-------------------------------------------------------------------------------------------------------------------------------------------

    def get_or_create_collection(self, collection_name):
        return self.client.get_or_create_collection(collection_name)
    
    def list_collections(self):
        return self.client.list_collections()
    
    def delete_collection(self, collection_name):
        return self.client.delete_collection(collection_name)
    
    def collection_chunk_count(self,collection_name):
        collection=self.get_or_create_collection(collection_name)
        return collection.count()

    def list_chunks_by_collection_name(self, collection_name):
        collection=self.get_or_create_collection(collection_name)
        return collection.get()

#Documents-------------------------------------------------------------------------------------------------------------------------------------------
    
    def document_chunk_count(self,collection_name, document_id):
        collection=self.get_or_create_collection(collection_name)
        result = collection.get(where={"DocumentId": {"$eq":document_id}})
        return len(result['ids'])

    # This function filters the chunks by document id in the metadata
    def list_chunks_by_document_id(self, collection_name, doc_id):
        collection=self.get_or_create_collection(collection_name)
        return collection.get(where={"DocumentId": {"$eq":doc_id}})

    def delete_document(self, collection_name, doc_id):
        collection=self.get_or_create_collection(collection_name)
        collection.delete(where={"DocumentId": {"$eq":doc_id}})


#ChunkGroup-------------------------------------------------------------------------------------------------------------------------------------------

    def delete_chunkgroup(self, collection_name, chunkgroup_id):
        collection=self.get_or_create_collection(collection_name)
        collection.delete(where={"ChunkGroupId": {"$eq":chunkgroup_id}})


#Chunks-------------------------------------------------------------------------------------------------------------------------------------------

    def get_chunk(self, collection_name, chunk_id):
        #collection=self.get_or_create_collection(collection_name)
        collection=self._get_collection(collection_name)
        chunk=collection.get(ids=[chunk_id])
        return chunk

    def delete_chunk(self, collection_name, chunk_id):
        collection=self.get_or_create_collection(collection_name)
        collection.delete(ids=[chunk_id])

#RAG-------------------------------------------------------------------------------------------------------------------------------------------

    # This version of get_collection includes the embedding function and is used for index and retrieval tasks
    # The distance function is also set to cosine
    def _get_collection(self, collection_name):
        if (self._ef is None):
            raise ValueError("ef is required")
        return self.client.get_or_create_collection(collection_name, embedding_function=self._ef, metadata={"hnsw:space": "cosine"})

    def index_chunk(self, 
                    collection_name, 
                    content, 
                    chunk_id, 
                    document_id, 
                    chunkgroup_id, 
                    source, 
                    abstract, 
                    title, 
                    published_date, 
                    keywords):
        if document_id is None:
            raise ValueError("document_id is required")
        if chunkgroup_id is None:
            raise ValueError("chunkgroup_id is required")
        try:
            metadata = {
                "DocumentId": document_id,
                "ChunkGroupId": chunkgroup_id,
                "Source": source if source else "",
                "Abstract": abstract if abstract else "",
                "Title": title if title else "",
                "PublishedDate": published_date if published_date else "",
                "Keywords": keywords if keywords else ""                
            }
            collection=self._get_collection(collection_name)
            collection.upsert(documents=[content],metadatas=[metadata],ids=[chunk_id])
        except Exception as e:
            logger.error(f"Failed to index chunk in chromadb: {e}, metadata={metadata}")
            raise e
        
    def retrieve(self, collection_name, query_texts, n_results=None):
        logger.info(f"Retrieving by query {query_texts}...")
        collection = self._get_collection(collection_name)
        if n_results is None:
            n_results = self.n_results
        result = collection.query(query_texts=query_texts, n_results=n_results)

        # Not found
        if 'ids' not in result or result['ids'] is None or len(result['ids']) == 0 or len(result['ids'][0]) == 0:
            return None

        if len(result['ids']) > 0:
            logger.debug('result='+ str(result['ids']))

        df = pd.DataFrame({
            'documents': result['documents'][0],
            'metadatas': result['metadatas'][0],
            'distances': result['distances'][0],
            'ids': result['ids'][0]
        })

        # drop duplicates
        return df.drop_duplicates(subset=['ids']).sort_values('distances', ascending=True)