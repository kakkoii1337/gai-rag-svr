import unittest
import os, sys

sys.path.insert(0,os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from gai.rag.server.dalc.IndexedDocumentChunk import IndexedDocumentChunk
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, selectinload, defer,joinedload, scoped_session
from gai.rag.server.dalc.Base import Base
from gai.lib.common.errors import DuplicatedDocumentException

from datetime import datetime
from gai.rag.server.dalc.RAGDBRepository import RAGDBRepository as Repository
from gai.rag.server.dalc.IndexedDocument import IndexedDocument
from gai.rag.server.dalc.IndexedDocumentChunkGroup import IndexedDocumentChunkGroup

from gai.lib.common.logging import getLogger
logger = getLogger(__name__)
from gai.lib.common import file_utils

import pytest

# Fixture to set up the database
def setup_database():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    return engine

def teardown_database(engine):
    engine.dispose()

#-------------------------------------------------------------------------------------------------------------------------------------------
        
def test_ut0011_convert_and_load_pdffile():
    # Arrange
    engine=setup_database()
    repo=Repository(engine)

    # Act
    file_path = os.path.join(os.path.dirname(__file__), "attention-is-all-you-need.pdf")
    file_type = file_path.split('.')[-1]
    content = repo._load_and_convert(file_path=file_path, file_type=file_type)

    # Assert
    assert content.startswith("3 2 0 2 g u A 2 ] L C . s c [ 7 v 2 6 7 3 0 . 6 0 7 1")
    teardown_database(engine)

def test_ut0012_convert_and_load_textfile():
    # Arrange
    engine=setup_database()
    repo=Repository(engine)
    
    # Act
    file_path = os.path.join(os.path.dirname(__file__), "pm_long_speech_2023.txt")
    content = repo._load_and_convert(file_path=file_path, file_type=file_path.split('.')[-1])

    # Assert
    assert content.startswith('PM Lee Hsien Loong delivered')
    teardown_database(engine)

#-------------------------------------------------------------------------------------------------------------------------------------------

def test_ut0013_create_document_hash():

    # Arrange
    engine=setup_database()
    repo=Repository(engine)
    file_path = os.path.join(os.path.dirname(__file__), "attention-is-all-you-need.pdf")

    # Act
    document_id = repo.create_document_hash(file_path)

    #convert to base 64
    assert "-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0"==document_id
    teardown_database(engine)

#-------------------------------------------------------------------------------------------------------------------------------------------
def test_ut0021_list_documents():

    # Arrange: Create 2 documents
    engine=setup_database()
    repo=Repository(engine)
    session = repo.session_factory()
    repo.create_document_header(
        collection_name='demo', 
        file_path=os.path.join(os.path.dirname(__file__), "attention-is-all-you-need.pdf"),
        file_type='pdf',
        session = session
    )
    repo.create_document_header(
        collection_name='demo1', 
        file_path=os.path.join(os.path.dirname(__file__), "pm_long_speech_2023.txt"),
        file_type='pdf',
        session=session
    )
    session.commit()

    # Act
    docs = repo.list_document_headers()

    # Assert
    assert len(docs)==2

    # Act
    docs = repo.list_document_headers(collection_name='demo1')

    # Assert
    assert len(docs)==1

if __name__=="__main__":
    pytest.main([sys.argv[0]])



