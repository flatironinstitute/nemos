from .abstract_solver import AbstractSolver, SolverState, StepResult
from typing import Generic, TypeVar, Type, Any, ClassVar
import inspect


class SolverAdapter(AbstractSolver[SolverState, StepResult]):
    _solver_cls: ClassVar[Type]
    _solver: Any

    def __getattr__(self, name: str):
        # without this guard deepcopy leads to a RecursionError
        try:
            solver = object.__getattribute__(self, "_solver")
        except AttributeError:
            raise AttributeError(name)

        return getattr(solver, name)

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        """Set of accepted argument names, extended with the wrapped solver's arguments."""
        own_arguments = set(inspect.getfullargspec(cls.__init__).args)
        solver_arguments = set(inspect.getfullargspec(cls._solver_cls.__init__).args)

        all_arguments = own_arguments | solver_arguments

        # discard arguments that are passed by BaseRegressor
        all_arguments.discard("self")
        all_arguments.discard("unregularized_loss")
        all_arguments.discard("regularizer")
        all_arguments.discard("regularizer_strength")

        return all_arguments

    def __init_subclass__(cls, **kw):
        """Generate the docstring including accepted arguments and the wrapped solver's documentation."""
        super().__init_subclass__(**kw)

        solver_cls = getattr(cls, "_solver_cls", None)
        if solver_cls is None:
            return

        # read the class's docstring or set it to a default
        adapter_doc = inspect.cleandoc(
            inspect.getdoc(cls) or f"Adapter for {solver_cls.__name__}"
        )

        # make a list of accepted arguments
        accepted_doc_header = inspect.cleandoc(
            f"""
            Accepted arguments:
            -------------------
            """
        )
        accepted_doc = "\n".join(f"- {a}" for a in sorted(cls.get_accepted_arguments()))
        accepted_doc = accepted_doc_header + "\n" + accepted_doc

        # read the underlying solver class's documentation
        solver_doc_header = inspect.cleandoc(
            f"""
            Solver's documentation:
            -----------------------
            """
        )
        solver_doc = inspect.cleandoc(
            inspect.getdoc(solver_cls) or "No class documentation found."
        )
        solver_doc = solver_doc_header + "\n" + solver_doc

        # read the underlying solver's __init__'s documentation
        solver_init_doc_header = inspect.cleandoc(
            f"""
            More info from {solver_cls.__name__}.__init__
            -----------------------------------------
            """
        )
        solver_init_doc = inspect.cleandoc(
            inspect.getdoc(solver_cls.__init__) or "No __init__ documentation found."
        )
        solver_init_doc = solver_init_doc_header + "\n" + solver_init_doc

        # the whole documentation is the parts after each other separated by blank lines
        class_doc = "\n\n".join(
            (
                adapter_doc,
                accepted_doc,
                solver_doc,
                solver_init_doc,
            )
        )

        cls.__doc__ = inspect.cleandoc(class_doc)
