# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Data Wrangler Environment.

The data_wrangler environment is a simple test environment that echoes back messages.
"""

from typing import Dict, List, Optional, Any
from openenv.core.env_server.types import Action, Observation # type: ignore
from pydantic import Field

class DataWranglerAction(Action):
    """Action for the Data Wrangler environment."""
    action_type: str = Field(..., description="Type of action: drop_column, rename_column, fill_missing, cast_type, extract_regex, datetime_parse, group_by_aggregate, submit")
    
    # Specifics depending on action_type
    target_column: Optional[str] = Field(None, description="The name of the column to act upon.")
    new_name: Optional[str] = Field(None, description="New name of the column (for rename_column/extract_regex).")
    fill_value: Optional[str] = Field(None, description="Value to fill missing data with.")
    cast_to: Optional[str] = Field(None, description="Target data type (for cast_type, e.g. 'int', 'float', 'datetime', 'string').")
    regex_pattern: Optional[str] = Field(None, description="Regex pattern for extracting data (for extract_regex).")
    format_string: Optional[str] = Field(None, description="Datetime format string (for datetime_parse, e.g., '%Y-%m-%d').")
    agg_column: Optional[str] = Field(None, description="Column to aggregate (for group_by_aggregate).")
    agg_func: Optional[str] = Field(None, description="Aggregation function (e.g., 'mean', 'sum', 'count').")

class DataWranglerObservation(Observation):
    """Observation representing the state of the dataset."""
    columns: List[str] = Field(default_factory=list, description="Current list of headers.")
    row_count: int = Field(default=0, description="Total number of rows in the dataset.")
    column_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Stats per column: dtype, missing_count, sample_values.")
    last_action_feedback: str = Field(default="Environment initialized.", description="Feedback from the last executed action.")
    is_done: bool = Field(default=False, description="Whether the task has terminated.")
