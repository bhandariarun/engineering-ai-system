from datetime import datetime, timezone
import ast
import operator


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def calculator(expression: str) -> str:
    """Evaluate a small arithmetic expression without executing arbitrary code."""
    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        raise ValueError("Only basic arithmetic is supported")

    tree = ast.parse(expression, mode="eval")
    return str(evaluate(tree))


def current_time() -> str:
    return datetime.now(timezone.utc).isoformat()


TOOLS = {"calculator": calculator, "current_time": current_time}
