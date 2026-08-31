"""Offline batch-comparison and analysis pipeline for AOI shadow-mode runs.

Scope: issue #6 (shadow-mode run + comparison analysis report), part of the
AOI-replacement epic (#1). This package does not depend on any real capture
or labeling system existing yet — it defines the schema, comparison math,
and report generation so it can be pointed at real logs once shadow mode is
physically running (blocked on issues #2/#3, hardware install).
"""
