import asyncio
import os, sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from gai.rag.server.gai_rag import RAG
from gai.lib.common.logging import getLogger
logger = getLogger(__name__)
import pytest
from gai.lib.common.utils import get_gai_config, get_app_path
import unittest

from rich.console import Console
console = Console()

@pytest.fixture(scope='module')
def rag():
    rag = RAG()
    return rag

def test_ut0031_rag_index(rag):

    # Arrange
    file_path = os.path.join(os.path.dirname(__file__), "attention-is-all-you-need.pdf")        
    rag.load()
    
    # Act
    try:
        doc_id = asyncio.run(
            rag.index_async(
            collection_name='demo', 
            file_path=file_path, 
            file_type='pdf',
            title='Attention is All You Need', 
            source='https://arxiv.org/abs/1706.03762', 
            authors='Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin',
            publisher = 'arXiv',
            published_date='2017-June-12', 
            comments='This is a test document',
            keywords=''
            ))
        print(doc_id)

        # Assert
        assert doc_id is not None
        assert doc_id == '-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0'
    except Exception as e:
        console.print(f"[red]Failed to index document: {e}[/]")
    finally:
        rag.unload()
    
def test_ut0032_rag_retrieve(rag):

    # Arrange
    rag.load()

    # Act
    try:
        result=rag.retrieve(
            collection_name='demo', 
            query_texts='What is the difference between transformer and RNN?',
            n_results=4
            )
        print(result)

        # Assert
    except Exception as e:
        console.print(f"[red]Failed to retrieve: {e}[/]")
    finally:
        rag.unload()


        

    