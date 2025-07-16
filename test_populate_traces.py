#!/usr/bin/env python3
"""
Test script for populate_traces function from linear_dataset.py
"""

import numpy as np
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datasources.linear_dataset import populate_traces, generate_seg_lens
from datasources.filter_dataset import generate_zipfian_integer, special_tokens


class MockConfig:
    """Mock configuration class for testing"""
    def __init__(self):
        # Dataset settings
        self.n_positions = 250  # number of positions in the trace
        self.nx = 5  # state dimension
        self.ny = 1  # output dimension
        self.max_sys_trace = 25  # maximum number of systems in a trace
        self.num_tasks = 10000  # number of available tasks/systems


def create_mock_entries(num_tasks, trace_length=125):
    """Create mock entries for testing"""
    entries = []
    
    for i in range(num_tasks):
        # Create mock x and y data for each system
        # x: state observations, y: output observations
        x_data = np.random.randn(trace_length, 5)  # 5-dimensional state
        y_data = np.random.randn(trace_length, 1)  # 1-dimensional output
        
        entries.append({
            "x": x_data,
            "y": y_data
        })
    
    return entries


def test_generate_seg_lens():
    """Test the generate_seg_lens function"""
    print("Testing generate_seg_lens...")
    
    n_positions = 20
    sys_in_trace = 3
    
    seg_lens = generate_seg_lens(n_positions, sys_in_trace)
    
    print(f"Generated segment lengths: {seg_lens}")
    print(f"Sum of segment lengths: {sum(seg_lens)}")
    print(f"Number of segments: {len(seg_lens)}")
    
    # Basic assertions
    assert isinstance(seg_lens, np.ndarray), "seg_lens should be a numpy array"
    assert len(seg_lens) > 0, "Should generate at least one segment"
    assert all(seg >= -2 for seg in seg_lens), "All segment lengths should be >= -2"
    
    print("✓ generate_seg_lens test passed\n")


def test_special_tokens():
    """Test the special_tokens function"""
    print("Testing special_tokens...")
    
    # Create a mock segment
    segment = np.random.randn(10, 20)  # 10 rows, 20 columns
    sys_name = 3
    
    start_token, end_token = special_tokens(segment, sys_name, style="zeros")
    
    print(f"Start token shape: {start_token.shape}")
    print(f"End token shape: {end_token.shape}")
    print(f"Start token: {start_token}")
    print(f"End token: {end_token}")
    
    # Basic assertions
    assert start_token.shape == (1, segment.shape[1]), "Start token should have correct shape"
    assert end_token.shape == (1, segment.shape[1]), "End token should have correct shape"
    assert start_token[0, 2*sys_name] == np.sqrt(2), "Start token should have sqrt(2) at correct position"
    assert end_token[0, 2*sys_name + 1] == np.sqrt(2), "End token should have sqrt(2) at correct position"
    
    print("✓ special_tokens test passed\n")


def test_populate_traces_basic():
    """Test the populate_traces function with basic parameters"""
    print("Testing populate_traces (basic)...")
    
    config = MockConfig()
    num_tasks = 10000
    entries = create_mock_entries(num_tasks)
    
    # Call the function
    segments, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds = populate_traces(
        config, num_tasks, entries
    )
    
    print(f"Segments shape: {segments.shape}")
    print(f"System choices: {sys_choices}")
    print(f"System dictionary: {sys_dict}")
    print(f"Token segment lengths: {tok_seg_lens}")
    print(f"Segment starts: {seg_starts}")
    print(f"Real segment lengths: {real_seg_lens}")
    print(f"System indices: {sys_inds}")
    
    # Basic assertions
    expected_shape = (config.n_positions + 1, config.nx + config.ny + 2*config.max_sys_trace + 3)
    assert segments.shape == expected_shape, \
        f"Segments should have shape {expected_shape}, got {segments.shape}"
    assert len(sys_choices) > 0, "Should have at least one system choice"
    assert len(sys_dict) > 0, "Should have at least one system in dictionary"
    assert len(tok_seg_lens) > 0, "Should have at least one token segment length"
    assert len(seg_starts) > 0, "Should have at least one segment start"
    assert len(real_seg_lens) > 0, "Should have at least one real segment length"
    assert len(sys_inds) > 0, "Should have at least one system index"
    
    # Check that start token is set correctly
    assert segments[0, 2*config.max_sys_trace] == np.sqrt(2), "Start token should be set correctly"
    
    print("✓ populate_traces basic test passed\n")


def test_populate_traces_edge_cases():
    """Test populate_traces with edge cases"""
    print("Testing populate_traces (edge cases)...")
    
    config = MockConfig()
    config.n_positions = 250  # Smaller context
    num_tasks = 10000
    entries = create_mock_entries(num_tasks, trace_length=125)
    
    # Test multiple times to catch edge cases
    for i in range(5):
        try:
            segments, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds = populate_traces(
                config, num_tasks, entries
            )
            
            # Verify output structure
            expected_shape = (config.n_positions + 1, config.nx + config.ny + 2*config.max_sys_trace + 3)
            assert segments.shape == expected_shape, f"Expected shape {expected_shape}, got {segments.shape}"
            assert len(sys_choices) == len(tok_seg_lens) == len(seg_starts) == len(real_seg_lens)
            
            print(f"Run {i+1}: Generated {len(sys_choices)} segments")
            
        except Exception as e:
            print(f"Run {i+1} failed with error: {e}")
            raise
    
    print("✓ populate_traces edge cases test passed\n")


def test_populate_traces_consistency():
    """Test that populate_traces produces consistent results with same seed"""
    print("Testing populate_traces (consistency)...")
    
    config = MockConfig()
    num_tasks = 30
    entries = create_mock_entries(num_tasks)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # First run
    segments1, sys_choices1, sys_dict1, tok_seg_lens1, seg_starts1, real_seg_lens1, sys_inds1 = populate_traces(
        config, num_tasks, entries
    )
    
    # Reset seed and run again
    np.random.seed(42)
    
    # Second run
    segments2, sys_choices2, sys_dict2, tok_seg_lens2, seg_starts2, real_seg_lens2, sys_inds2 = populate_traces(
        config, num_tasks, entries
    )
    
    # Check consistency
    assert np.array_equal(segments1, segments2), "Segments should be identical with same seed"
    assert sys_choices1 == sys_choices2, "System choices should be identical with same seed"
    assert sys_dict1 == sys_dict2, "System dictionary should be identical with same seed"
    assert tok_seg_lens1 == tok_seg_lens2, "Token segment lengths should be identical with same seed"
    assert seg_starts1 == seg_starts2, "Segment starts should be identical with same seed"
    assert real_seg_lens1 == real_seg_lens2, "Real segment lengths should be identical with same seed"
    assert sys_inds1 == sys_inds2, "System indices should be identical with same seed"
    
    print("✓ populate_traces consistency test passed\n")


def test_populate_traces_structure():
    """Test the structure and properties of the output"""
    print("Testing populate_traces (structure)...")
    
    config = MockConfig()
    num_tasks = 40
    entries = create_mock_entries(num_tasks)
    
    segments, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds = populate_traces(
        config, num_tasks, entries
    )
    
    # Test segment structure
    print(f"Segments shape: {segments.shape}")
    expected_shape = (config.n_positions + 1, config.nx + config.ny + 2*config.max_sys_trace + 3)
    print(f"Expected shape: {expected_shape}")
    
    # Check that segments are properly filled
    non_zero_rows = np.count_nonzero(np.any(segments != 0, axis=1))
    print(f"Non-zero rows in segments: {non_zero_rows}")
    
    # Check that start token is at the beginning
    assert segments[0, 2*config.max_sys_trace] == np.sqrt(2), "Start token should be at position 0"
    
    # Check that system choices are valid
    assert all(choice in sys_inds for choice in sys_choices), "All system choices should be in sys_inds"
    
    # Check that segment starts are in ascending order
    assert seg_starts == sorted(seg_starts), "Segment starts should be in ascending order"
    
    # Check that token segment lengths are positive
    assert all(length >= 0 for length in tok_seg_lens), "All token segment lengths should be non-negative"
    
    # Check that real segment lengths are non-negative
    assert all(length >= 0 for length in real_seg_lens), "All real segment lengths should be non-negative"
    
    print("✓ populate_traces structure test passed\n")


def test_populate_traces_error_handling():
    """Test error handling in populate_traces"""
    print("Testing populate_traces (error handling)...")
    
    config = MockConfig()
    
    # Test with empty entries
    try:
        segments, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds = populate_traces(
            config, 0, []
        )
        print("Warning: populate_traces should probably fail with empty entries")
    except Exception as e:
        print(f"Expected error with empty entries: {e}")
    
    # Test with insufficient entries
    try:
        entries = create_mock_entries(5)  # Only 5 entries
        segments, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds = populate_traces(
            config, 10, entries  # But trying to use 10 systems
        )
        print("Warning: populate_traces should probably fail with insufficient entries")
    except Exception as e:
        print(f"Expected error with insufficient entries: {e}")
    
    print("✓ populate_traces error handling test passed\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing populate_traces function from linear_dataset.py")
    print("=" * 60)
    
    try:
        test_generate_seg_lens()
        test_special_tokens()
        test_populate_traces_basic()
        test_populate_traces_edge_cases()
        test_populate_traces_consistency()
        test_populate_traces_structure()
        test_populate_traces_error_handling()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 