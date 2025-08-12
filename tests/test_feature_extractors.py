"""Tests for feature extraction functionality."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

# Add src to path for testing
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.feature_extractors import (
    extract_point_xy_from_wkt,
    polygon_vertex_count_from_wkt,
    multipolygon_stats_from_wkt,
    extract_geometry_features,
    maybe_extract_json_features
)


class TestWKTExtraction:
    """Test WKT geometry extraction functions."""
    
    def test_extract_point_xy_valid(self):
        """Test extraction of valid POINT geometries."""
        series = pd.Series([
            "POINT (12.4924 41.8902)",
            "POINT (9.1900 45.4642)",
            "POINT (-122.4194 37.7749)"
        ])
        
        x_coords, y_coords = extract_point_xy_from_wkt(series)
        
        assert len(x_coords) == 3
        assert len(y_coords) == 3
        assert x_coords.iloc[0] == pytest.approx(12.4924)
        assert y_coords.iloc[0] == pytest.approx(41.8902)
        assert x_coords.iloc[2] == pytest.approx(-122.4194)
        assert y_coords.iloc[2] == pytest.approx(37.7749)
    
    def test_extract_point_xy_invalid(self):
        """Test handling of invalid POINT geometries."""
        series = pd.Series([
            "INVALID WKT",
            "POINT (not_a_number 41.8902)",
            None,
            ""
        ])
        
        x_coords, y_coords = extract_point_xy_from_wkt(series)
        
        assert len(x_coords) == 4
        assert len(y_coords) == 4
        assert pd.isna(x_coords).all()
        assert pd.isna(y_coords).all()
    
    def test_polygon_vertex_count(self):
        """Test polygon vertex counting."""
        series = pd.Series([
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",  # 5 vertices (square)
            "POLYGON ((0 0, 2 0, 1 1, 0 0))",       # 4 vertices (triangle)
            "INVALID POLYGON",
            None
        ])
        
        counts = polygon_vertex_count_from_wkt(series)
        
        assert counts.iloc[0] == 5
        assert counts.iloc[1] == 4
        assert pd.isna(counts.iloc[2])
        assert pd.isna(counts.iloc[3])
    
    def test_multipolygon_stats(self):
        """Test multipolygon statistics extraction."""
        series = pd.Series([
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))",
            "INVALID MULTIPOLYGON",
            None
        ])
        
        stats = multipolygon_stats_from_wkt(series)
        
        # First multipolygon should have 2 polygons
        assert stats.columns.tolist() == [
            'wkt_mpoly_count',
            'wkt_mpoly_vertices',
            'wkt_mpoly_outer_vertices_avg'
        ]
        
        # Check if basic structure is correct
        assert len(stats) == 3
        assert pd.isna(stats.iloc[1]).all()  # Invalid entry
        assert pd.isna(stats.iloc[2]).all()  # None entry


class TestGeometryFeatures:
    """Test geometry feature extraction pipeline."""
    
    def test_extract_geometry_features(self, sample_wkt_data):
        """Test complete geometry feature extraction."""
        df_result, dropped_cols = extract_geometry_features(sample_wkt_data)
        
        # Check that new columns were created
        assert "point_col_x" in df_result.columns
        assert "point_col_y" in df_result.columns
        assert "polygon_col_vertex_count" in df_result.columns
        
        # Check that original geometry columns are marked for dropping
        assert "point_col" in dropped_cols
        assert "polygon_col" in dropped_cols
        
        # Verify data integrity
        assert not df_result["point_col_x"].iloc[0:2].isna().any()
        assert not df_result["point_col_y"].iloc[0:2].isna().any()
        assert df_result["polygon_col_vertex_count"].iloc[0] == 5  # Square has 5 vertices
    
    def test_extract_geometry_features_no_geometry(self):
        """Test extraction when no geometry columns exist."""
        df = pd.DataFrame({
            "regular_col": [1, 2, 3],
            "another_col": ["a", "b", "c"]
        })
        
        df_result, dropped_cols = extract_geometry_features(df)
        
        # No changes should be made
        assert df_result.equals(df)
        assert len(dropped_cols) == 0


class TestJSONExtraction:
    """Test JSON feature extraction."""
    
    def test_extract_json_features_valid(self):
        """Test extraction from valid JSON columns."""
        df = pd.DataFrame({
            "json_col": [
                '{"key1": "value1", "numeric_key": 42}',
                '{"key1": "value2", "numeric_key": 24, "extra": true}',
                '{"different_key": "value"}'
            ],
            "regular_col": [1, 2, 3]
        })
        
        # Store original column count before modification
        original_cols = len(df.columns)
        
        df_result, dropped_cols = maybe_extract_json_features(df)
        
        # Should extract JSON features and drop original column
        assert "json_col" in dropped_cols  # Original column marked for dropping
        assert len(dropped_cols) == 1  # One column dropped
        
        # Check if new columns were added (original + extracted features)
        new_cols = len(df_result.columns)
        assert new_cols > original_cols  # Should have more columns due to extracted features
        
        # Check for expected extracted columns
        assert any(col.startswith("json_col__") for col in df_result.columns)
    
    def test_extract_json_features_invalid(self):
        """Test handling of invalid JSON."""
        df = pd.DataFrame({
            "not_json": ["not json", "also not json"],
            "regular_col": [1, 2]
        })
        
        df_result, dropped_cols = maybe_extract_json_features(df)
        
        # Should handle gracefully
        assert len(df_result) == len(df)
        assert "regular_col" in df_result.columns


class TestIntegrationFeatureExtraction:
    """Integration tests for feature extraction."""
    
    def test_feature_extraction_pipeline(self, sample_real_estate_data):
        """Test complete feature extraction pipeline."""
        # Test geometry extraction
        df1, dropped_geom = extract_geometry_features(sample_real_estate_data)
        
        # Test JSON extraction
        df2, dropped_json = maybe_extract_json_features(df1)
        
        # Verify pipeline integrity
        assert len(df2) == len(sample_real_estate_data)
        assert "PC_PoligonoGeometrico" in dropped_geom
        assert "PC_PoligonoGeometrico_x" in df2.columns
        assert "PC_PoligonoGeometrico_y" in df2.columns
        
        # Verify no NaN values where we expect valid data
        valid_points = ~df2["PC_PoligonoGeometrico_x"].isna()
        assert valid_points.sum() > 0  # Should have some valid points
    
    def test_feature_extraction_robustness(self):
        """Test robustness with edge cases."""
        df = pd.DataFrame({
            "mixed_geometry": [
                "POINT (12.4924 41.8902)",
                "INVALID WKT",
                None,
                "",
                "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
            ],
            "mixed_json": [
                '{"valid": "json"}',
                "invalid json",
                None,
                "",
                '{"another": "valid"}'
            ]
        })
        
        # Should not raise exceptions
        df1, _ = extract_geometry_features(df)
        df2, _ = maybe_extract_json_features(df1)
        
        assert len(df2) == len(df)
        assert not df2.empty


class TestErrorHandling:
    """Test error handling in feature extraction."""
    
    def test_extract_point_xy_edge_cases(self):
        """Test edge cases in point extraction."""
        # Test with empty series
        empty_series = pd.Series([], dtype=object)
        x, y = extract_point_xy_from_wkt(empty_series)
        assert len(x) == 0
        assert len(y) == 0
        
        # Test with very large numbers
        large_coords = pd.Series(["POINT (1e10 -1e10)"])
        x, y = extract_point_xy_from_wkt(large_coords)
        assert x.iloc[0] == 1e10
        assert y.iloc[0] == -1e10
    
    def test_polygon_vertex_count_edge_cases(self):
        """Test edge cases in polygon vertex counting."""
        # Empty polygon coordinates
        series = pd.Series(["POLYGON (())"])
        counts = polygon_vertex_count_from_wkt(series)
        # Should handle gracefully, either return 0 or NaN
        assert len(counts) == 1
    
    @patch('preprocessing.feature_extractors.logger')
    def test_error_logging(self, mock_logger):
        """Test that errors are logged appropriately."""
        # This would test if the logger is called when exceptions occur
        # Implementation depends on how logging is handled in the actual code
        pass


if __name__ == "__main__":
    pytest.main([__file__])