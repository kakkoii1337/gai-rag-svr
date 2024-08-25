from gai.rag.server.gai_rag import RAG
from gai.lib.common.logging import getLogger
logger = getLogger(__name__)
import unittest
import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__), "..", "..", ".."))

class UT0040_RAG_APITest(unittest.TestCase):

    def test_in_memory_if_env_true(self):
        os.environ["IN_MEMORY"] = "True"
        from gai.rag.server.api.main import get_in_memory
        in_memory = get_in_memory()
        self.assertEqual(in_memory, True)

    def test_in_memory_if_env_false(self):
        os.environ["IN_MEMORY"] = "False"
        from gai.rag.server.api.main import get_in_memory
        in_memory = get_in_memory()
        self.assertEqual(in_memory, False)

    def test_in_memory_if_env_none(self):
        os.environ.pop("IN_MEMORY", None)
        from gai.rag.server.api.main import get_in_memory
        in_memory = get_in_memory()
        self.assertEqual(in_memory, True)

if __name__ == "__main__":
    unittest.main()