"""Plugin for supporting the spaCy's Extension (._.) attribute"""
# import functools
# from typing import Iterable, List, cast, Callable, Union
# from typing_extensions import Final, Literal
#
# import mypy.plugin  # To avoid circular imports.
# from mypy.checker import TypeChecker
# from mypy.plugin import AttributeContext, MethodContext
# from mypy.types import Type, CallableType
# from mypy.plugin import Plugin
#
# from ttc.language.russian.token_extensions import TOKEN_EXTENSIONS
# from ttc.language.russian.span_extensions import SPAN_EXTENSIONS
# from inspect import signature
#
# PREFIX = "spacy.tokens.underscore.Underscore."
#
# TOKEN_EXTENSIONS = {
#     PREFIX + name: TOKEN_EXTENSIONS[name] for name in list(TOKEN_EXTENSIONS)
# }
# SPAN_EXTENSIONS = {
#     PREFIX + name: SPAN_EXTENSIONS[name] for name in list(SPAN_EXTENSIONS)
# }
#
#
# def span_extension_hook(
#     fullname: str,
#     ctx: Union[AttributeContext, MethodContext],
# ) -> Type:
#     assert isinstance(ctx.api, TypeChecker)
#     assert isinstance(ctx.context.name, str)
#
#     fn_sig = signature(SPAN_EXTENSIONS[fullname])
#     if t := fn_sig.return_annotation:
#         t: type
#         breakpoint()
#         if len(fn_sig.parameters) > 1:
#             return CallableType(arg_types=[], ret_type=t)
#         return t
#     elif any(ctx.context.name.startswith(p) for p in ("is_", "has_")):
#         return ctx.api.named_type("builtins.bool")
#
#     return ctx.default_return_type
#
#
# def token_extension_hook(
#     fullname: str,
#     ctx: Union[AttributeContext, MethodContext],
# ) -> Type:
#     assert isinstance(ctx.api, TypeChecker)
#     assert isinstance(ctx.context.name, str)
#
#     if t := fn_sig.return_annotation:
#         t: type
#         breakpoint()
#         if len(fn_sig.parameters) > 1:
#             return CallableType(arg_types=[], ret_type=t)
#         return t
#     elif any(ctx.context.name.startswith(p) for p in ("is_", "has_")):
#         return ctx.api.named_type("builtins.bool")
#
#     return ctx.default_return_type
#
#
# class SpacyExtensionAttributePlugin(Plugin):
#     def get_attribute_hook(self, fullname: str):
#         if fullname in TOKEN_EXTENSIONS:
#             print(fullname, "<<< Token ext attr")
#             return functools.partial(token_extension_hook, fullname)
#         if fullname in SPAN_EXTENSIONS:
#             print(fullname, "<<< Span ext attr")
#             return functools.partial(span_extension_hook, fullname)
#         return None
#
#     def get_method_hook(self, fullname: str):
#         if fullname in TOKEN_EXTENSIONS:
#             print(fullname, "<<< Token ext method")
#             return functools.partial(token_extension_hook, fullname)
#         if fullname in SPAN_EXTENSIONS:
#             print(fullname, "<<< Span ext method")
#             return functools.partial(span_extension_hook, fullname)
#         return None
#
#
# def plugin(version: str):
#     return SpacyExtensionAttributePlugin
