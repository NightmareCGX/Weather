"""Shared FastAPI dependencies for the API service."""

from dataclasses import dataclass
from typing import Any

from fastapi import Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import Select
from sqlalchemy.orm import Session

__all__ = ["Page", "PaginationParams", "paginate"]


class PaginationParams(BaseModel):
    """Cursor pagination parameters (API.md section 2.5).

    ``limit`` defaults to 20 with a maximum of 100. ``starting_after`` and
    ``ending_before`` are mutually exclusive; providing both is rejected with a
    422 since API.md does not define a precedence.
    """

    limit: int = Field(default=20, ge=1, le=100)
    starting_after: str | None = Field(default=None)
    ending_before: str | None = Field(default=None)


async def pagination_params(
    limit: int = Query(default=20, ge=1, le=100),
    starting_after: str | None = Query(default=None),
    ending_before: str | None = Query(default=None),
) -> PaginationParams:
    """Resolve and validate the pagination query parameters for a request.

    Parameters are read as individual query fields so that invalid ``limit``
    values are reported as validation errors by FastAPI, and the mutually
    exclusive cursor rule is enforced with an explicit HTTP 422.
    """
    if starting_after is not None and ending_before is not None:
        raise HTTPException(
            status_code=422,
            detail="starting_after and ending_before are mutually exclusive",
        )
    return PaginationParams(
        limit=limit,
        starting_after=starting_after,
        ending_before=ending_before,
    )


Pagination = Depends(pagination_params)


@dataclass
class Page:
    """A single page of results and its pagination metadata."""

    items: list[Any]
    has_more: bool
    next_cursor: str | None


def _cursor_value(item: Any, cursor_column: Any) -> str:
    return str(getattr(item, cursor_column.key))


def paginate(
    db: Session,
    stmt: Select[Any],
    cursor_column: Any,
    pagination: PaginationParams,
) -> Page:
    """Apply cursor pagination to a statement and return one page.

    The cursor is the resource ``id`` (a unique column in every catalog table),
    enabling a single uniform pattern across all catalog endpoints. Forward
    pages order ascending and return the last item's id as ``next_cursor`` when
    more rows exist; backward pages order descending and return the first
    item's id so the client can continue with ``ending_before``.
    """
    limit = pagination.limit
    descending = pagination.ending_before is not None

    if pagination.starting_after is not None:
        stmt = stmt.where(cursor_column > pagination.starting_after)
    elif pagination.ending_before is not None:
        stmt = stmt.where(cursor_column < pagination.ending_before)

    if descending:
        stmt = stmt.order_by(cursor_column.desc())
    else:
        stmt = stmt.order_by(cursor_column.asc())
    stmt = stmt.limit(limit + 1)

    rows = list(db.execute(stmt).scalars().all())
    has_more = len(rows) > limit
    items = rows[:limit]
    if descending:
        items = list(reversed(items))

    if has_more and items:
        boundary = items[0] if descending else items[-1]
        next_cursor = _cursor_value(boundary, cursor_column)
    else:
        next_cursor = None

    return Page(items=items, has_more=has_more, next_cursor=next_cursor)
