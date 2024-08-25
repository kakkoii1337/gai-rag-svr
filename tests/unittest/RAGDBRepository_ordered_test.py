import unittest
import os, sys

sys.path.insert(0,os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from gai.rag.server.dalc.IndexedDocumentChunk import IndexedDocumentChunk
from sqlalchemy import create_engine
from sqlalchemy.orm import selectinload, defer
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
@pytest.fixture(scope='module')
def engine():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()

@pytest.fixture(scope='module')
def repo(engine):
    return Repository(engine)

#-------------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.run(order=1)
def test_ot01_create_document_header(repo):

    # Arrange
    session = repo.session_factory()
    file_path = os.path.join(os.path.dirname(__file__), "attention-is-all-you-need.pdf")

    # Act: create_document_header()
    
    document_id = repo.create_document_header(
        collection_name='demo', 
        file_path=file_path, 
        file_type='pdf',
        title='Attention is All You Need', 
        source='https://arxiv.org/abs/1706.03762', 
        abstract="""The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data""",
        authors='Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin',
        publisher = 'arXiv',
        published_date='2017-June-12', 
        comments='This is a test document',
        session=session)

    # Assert

    assert document_id == "-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0"

    retrieved_header = session.query(IndexedDocument).options(
                    selectinload(IndexedDocument.ChunkGroups),
                    defer(IndexedDocument.File)
                ).filter(
                    IndexedDocument.Id=="-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0", 
                    IndexedDocument.CollectionName=="demo"
                ).first()

    # Ensure the document was retrieved
    assert retrieved_header is not None

    # Check each field for correctness
    assert retrieved_header.CollectionName=='demo'
    assert os.path.basename(retrieved_header.FileName) == os.path.basename(file_path)
    assert retrieved_header.ByteSize == os.path.getsize(file_path)
    assert retrieved_header.FileType == 'pdf'
    assert retrieved_header.Title == 'Attention is All You Need'
    assert retrieved_header.Source == 'https://arxiv.org/abs/1706.03762'
    assert retrieved_header.Abstract.strip() == (
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. "
        "The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, "
        "the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation "
        "tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves "
        "28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the "
        "WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days "
        "on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other "
        "tasks by applying it successfully to English constituency parsing both with large and limited training data"
    ).strip()
    assert retrieved_header.Authors == 'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin'
    assert retrieved_header.Publisher == 'arXiv'
    assert retrieved_header.PublishedDate == datetime.strptime('2017-June-12','%Y-%B-%d').date()
    assert retrieved_header.Comments == 'This is a test document'

    # Additional checks for fields with default values or calculated fields
    assert retrieved_header.IsActive is True
    assert retrieved_header.CreatedAt is not None
    assert retrieved_header.UpdatedAt is not None

    # Act: get_document_header()

    retrieved_header = repo.get_document_header(collection_name='demo', document_id="-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0")

    # Assert

    retrieved_header = session.query(IndexedDocument).options(
                    selectinload(IndexedDocument.ChunkGroups),
                    defer(IndexedDocument.File)
                ).filter(
                    IndexedDocument.Id=="-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0", 
                    IndexedDocument.CollectionName=="demo"
                ).first()

    # Ensure the document was retrieved
    assert retrieved_header is not None

    # Check each field for correctness
    assert retrieved_header.CollectionName=='demo'
    assert os.path.basename(retrieved_header.FileName) == os.path.basename(file_path)
    assert retrieved_header.ByteSize == os.path.getsize(file_path)
    assert retrieved_header.FileType == 'pdf'
    assert retrieved_header.Title == 'Attention is All You Need'
    assert retrieved_header.Source == 'https://arxiv.org/abs/1706.03762'
    assert retrieved_header.Abstract.strip() == (
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. "
        "The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, "
        "the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation "
        "tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves "
        "28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the "
        "WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days "
        "on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other "
        "tasks by applying it successfully to English constituency parsing both with large and limited training data"
    ).strip()
    assert retrieved_header.Authors == 'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin'
    assert retrieved_header.Publisher == 'arXiv'
    assert retrieved_header.PublishedDate == datetime.strptime('2017-June-12','%Y-%B-%d').date()
    assert retrieved_header.Comments == 'This is a test document'

    # Additional checks for fields with default values or calculated fields
    assert retrieved_header.IsActive is True
    assert retrieved_header.CreatedAt is not None
    assert retrieved_header.UpdatedAt is not None    

#-------------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.run(order=2)
def test_ot02_should_not_allow_duplicate(repo):
    # Arrange
    file_path = os.path.join(os.path.dirname(__file__), "attention-is-all-you-need.pdf")

    # Act
    try:
        repo.create_document_header(
            collection_name='demo', 
            file_path=file_path, 
            file_type='pdf',
            title='Attention is All You Need', 
            source='https://arxiv.org/abs/1706.03762', 
            abstract="""The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data""",
            authors='Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin',
            publisher = 'arXiv',
            published_date='2017-June-12', 
            comments='This is a test document')
    except DuplicatedDocumentException as e:
        assert e.code == "duplicate_document"

#-------------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.run(order=3)
def test_ot03_create_chunk_group_with_external_session(repo):
    # Arrange
    session = repo.session_factory()

    # Act: Use internal session for act
    chunkgroup = repo.create_chunkgroup(
        document_id='-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0', 
        collection_name='demo',
        chunk_size=1000, 
        chunk_overlap=100, 
        splitter=file_utils.split_text,
        session=session)
    session.commit()

    # Assert: Use external session to assert
    retrieved_doc = session.query(IndexedDocument).filter_by(Id='-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0').first()
    retrieved_group = session.query(IndexedDocumentChunkGroup).filter_by(Id=chunkgroup.Id).first()

    # Assert: Can get group from doc
    assert len(retrieved_doc.ChunkGroups) == 1

    # Assert: Can get group from DB
    assert retrieved_group is not None

    # Assert: Chunk group has correct number of chunks
    file_chunks = os.listdir(retrieved_group.ChunksDir)
    assert retrieved_group.ChunkCount== len(file_chunks)

#-------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.run(order=4)
def test_ot04_create_chunks(repo):
    session = repo.session_factory()

    # Arrange
    header = repo.get_document_header(collection_name='demo',document_id='-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0')
    group = header.ChunkGroups[0]

    #Act
    chunks = repo.create_chunks(group.Id,group.ChunksDir,session=session)
    session.commit()

    # Assert

    # List all chunks
    #retrieved_chunks = repo.list_chunks()
    retrieved_chunks = session.query(IndexedDocumentChunk).all()
    assert len(chunks)==len(retrieved_chunks)
    for i in range(len(retrieved_chunks)):
        assert retrieved_chunks[i].Id == chunks[i].Id

    # List only chunks by id
    retrieved_chunks = repo.list_chunks_by_document_id(document_id='-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0')
    assert len(chunks) == len(retrieved_chunks)
    for i in range(len(retrieved_chunks)):
        assert retrieved_chunks[i].Id == chunks[i].Id


#-------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.run(order=5)
def test_ot05_delete_document(repo):

    # Act
    repo.delete_document(collection_name='demo',document_id='-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0')

    # Assert
    session = repo.session_factory()
    retrieved_doc = session.query(IndexedDocument).filter_by(Id='-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0').first()
    assert retrieved_doc is None

    retrieved_group = session.query(IndexedDocumentChunkGroup).filter_by(DocumentId='-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0').first()
    assert retrieved_group is None

    retrieved_chunks = session.query(IndexedDocumentChunk).all()
    assert len(retrieved_chunks) == 0

#-------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.run(order=6)
def test_ot06_delete_collection(repo):

    # Arrange: Recreate the document for deletion
    session = repo.session_factory()
    document_id=repo.create_document_header(
        collection_name='demo', 
        file_path=os.path.join(os.path.dirname(__file__), "attention-is-all-you-need.pdf"),
        file_type='pdf',
        session = session
    )
    group = repo.create_chunkgroup(
        collection_name='demo',
        document_id=document_id, 
        chunk_size=1000, 
        chunk_overlap=100, 
        splitter=file_utils.split_text, 
        session=session)
    repo.create_chunks(
        group.Id,
        group.ChunksDir,
        session=session)
    session.commit()

    # Act
    repo.delete_collection('demo')

    # Assert
    session=repo.session_factory()
    retrieved_doc = session.query(IndexedDocument).filter(
        IndexedDocument.Id=='-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0',
        IndexedDocument.CollectionName=='demo').first()
    assert retrieved_doc is None

    retrieved_group = session.query(IndexedDocumentChunkGroup).filter_by(DocumentId='-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0').first()
    assert retrieved_group is None

    retrieved_chunks = session.query(IndexedDocumentChunk).all()
    assert len(retrieved_chunks) == 0


if __name__=="__main__":
    pytest.main([sys.argv[0]])



