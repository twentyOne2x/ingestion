from .parent import ParentNode
from .child import ChildNode

PARENT_JSON_SCHEMA = ParentNode.model_json_schema()
CHILD_JSON_SCHEMA = ChildNode.model_json_schema()
