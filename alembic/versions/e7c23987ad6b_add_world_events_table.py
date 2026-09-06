"""add world_events table

Revision ID: e7c23987ad6b
Revises: 3b7f2a91c4d6
Create Date: 2026-08-21 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from geoalchemy2 import Geometry

# revision identifiers, used by Alembic.
revision: str = 'e7c23987ad6b'
down_revision: Union[str, Sequence[str], None] = '3b7f2a91c4d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'world_events',
        sa.Column('id', sa.String(length=20), nullable=False),
        sa.Column('category', sa.String(length=20), nullable=False),
        sa.Column('event_code', sa.String(length=10), nullable=True),
        sa.Column('actor1_name', sa.Text(), nullable=True),
        sa.Column('actor2_name', sa.Text(), nullable=True),
        sa.Column('action_geo_full_name', sa.Text(), nullable=True),
        sa.Column('lat', sa.REAL(), nullable=True),
        sa.Column('lon', sa.REAL(), nullable=True),
        sa.Column(
            'geom',
            Geometry(geometry_type='POINT', srid=4326, dimension=2, spatial_index=False, from_text='ST_GeomFromEWKT', name='geometry'),
            nullable=True,
        ),
        sa.Column('event_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('num_mentions', sa.Integer(), nullable=True),
        sa.Column('num_sources', sa.Integer(), nullable=True),
        sa.Column('goldstein_scale', sa.REAL(), nullable=True),
        sa.Column('avg_tone', sa.REAL(), nullable=True),
        sa.Column('source_url', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_world_events_event_date', 'world_events', ['event_date'], unique=False)
    op.create_index('idx_world_events_geom', 'world_events', ['geom'], unique=False, postgresql_using='gist')


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_world_events_geom', table_name='world_events', postgresql_using='gist')
    op.drop_index('idx_world_events_event_date', table_name='world_events')
    op.drop_table('world_events')
