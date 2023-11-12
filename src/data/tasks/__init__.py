import pkgutil
import importlib

# Discover and import dynamically all modules in the package
package = __name__
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    importlib.import_module(f"{package}.{module_name}")
