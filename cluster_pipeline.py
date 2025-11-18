"""Backward-compatible entry point for the expedition_clustering package.

The original notebooks imported classes/functions from this module.  After
converting the repository into an installable package we simply re-export the
public API so older notebooks/scripts keep working unchanged.
"""

from expedition_clustering.pipeline import *  # noqa: F401,F403
