# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Shared code for all scripts and modules """
import logging

__version__ = "2022.10.13"
log = logging.getLogger("dlab.core")


def get_or_verify_datafile(resource, skip_verify=False):
    """Load a neurobank resource, either from a local file or from an archive

    if `resource` is a file that exists, the hash is used to look up resource information
    if `resource` is an ID or a registry URL, tries to find the file in a local archive

    If `skip_verify` is True, does the lookup by the name of the file (to avoid long hashes)

    returns (datafile path, registry record)
    """
    from pathlib import Path
    import nbank

    p = Path(resource)
    if p.exists():
        datafile = p
        if skip_verify:
            log.info(" - using local file %s", datafile)
            name = p.stem
            resource_info = nbank.describe(name)
        else:
            log.info(" - using local file %s (checking hash)", datafile)
            resources = nbank.verify(datafile)
            try:
                resource_info = next(resources)
                log.info("   ✓ %s", resource_info["sha1"])
            except StopIteration:
                raise ValueError(
                    "sha1 for local file does not match any resource in the registry"
                )
    else:
        resource_url = nbank.full_url(resource)
        log.info(" - source resource: %s", resource_url)
        resource_info = nbank.describe(resource_url)
        datafile = nbank.get(resource, local_only=True)
        if datafile is None:
            raise ValueError("unable to locate resource file")
    return datafile, resource_info


def fetch_resource(neurobank_registry, resource_name):
    """Fetch a downloadable resource from the registry.

    The file will be cached locally. Returns the path of the file.
    """
    from pathlib import Path
    from appdirs import user_cache_dir
    from urllib.parse import urlparse
    from nbank import fetch_resource

    APP_NAME = "dlab"
    APP_AUTHOR = "melizalab"
    parsed_url = urlparse(neurobank_registry)
    cache_dir = Path(user_cache_dir(APP_NAME, APP_AUTHOR)) / parsed_url.netloc
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / resource_name
    if not target.exists():
        log.debug("   - fetching %s from registry", resource_name)
        fetch_resource(neurobank_registry, resource_name, target)
    return str(target)
