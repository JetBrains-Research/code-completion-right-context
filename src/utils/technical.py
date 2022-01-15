import inspect

from os import path
from typeguard import typechecked
from importlib.machinery import SourceFileLoader


def load_module(script_path):
    """
    Import module from script_path.

    Parameters
    ----------
    script_path : str
        Path to script to import.

    Returns
    -------
    model_script : module
    """
    script_name_with_ext = path.split(script_path)[1]
    script_name_without_ext = path.splitext(script_name_with_ext)[0]
    return SourceFileLoader(script_name_without_ext, script_path).load_module()


def annotations_from_parent(child_class):
    """
    Decorator to get annotations from parent class.
    Annotations are got from the nearest parent
    in the inheritance hierarchy.

    If child method has its own annotations it is skipped.
    """
    parent_class = child_class.__mro__[1]

    for method_name in dir(parent_class):

        parent_method = getattr(parent_class, method_name)
        if not hasattr(parent_method, '__annotations__') or parent_method.__annotations__ == {}:
            continue
        child_method = getattr(child_class, method_name)

        # skip if child method has its own annotations
        if getattr(child_method, '__annotations__', None):
            continue

        # hook for typeguard package
        if hasattr(parent_method, '__wrapped__'):
            typeguard_mode = True
            parent_method = parent_method.__wrapped__
            # if decorator is used then there are no annotations, but better to check
            if hasattr(child_method, '__wrapped__'):
                raise TypeError('Unespected annotations in child method')
        else:
            typeguard_mode = False

        # check that varnames are equal
        parent_method_varnames = inspect.getargs(parent_method.__code__).args
        child_method_varnames = inspect.getargs(child_method.__code__).args
        if parent_method_varnames != child_method_varnames:
            assertion_message = (
                f"different varnames {method_name}: \n"
                f"parent varnames: {parent_method_varnames}, \n"
                f"child varnames: {child_method_varnames}, \n"
                f"parent method {parent_method}, \n"
                f"child method {child_method}, \n"
                f"parent method has wrap {hasattr(parent_method, '__wrapped__')}, \n"
                f"child method has wrap {hasattr(child_method, '__wrapped__')}. \n"
            )
            raise TypeError(assertion_message)

        child_method.__annotations__ = parent_method.__annotations__

        if typeguard_mode:
            wrapped_child_method = typechecked(child_method)
            setattr(child_class, method_name, wrapped_child_method)

    return child_class

