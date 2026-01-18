"""
Tests for the Data Assimilation Base module.

Tests include functionality for the base classes and ensemble generator.
"""
import pytest
import torch
from torch_geometric.data import Data
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Use direct import to avoid package conflicts
import importlib.util
import sys
import os

# Add the graph_weather directory to the path to make relative imports work
sys.path.insert(0, os.path.join(os.getcwd(), 'graph_weather'))

spec = importlib.util.spec_from_file_location('data_assimilation_base', './graph_weather/models/data_assimilation/data_assimilation_base.py')
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)

DataAssimilationBase = base_module.DataAssimilationBase
EnsembleGenerator = base_module.EnsembleGenerator


def test_data_assimilation_base_initialization():
    """Test DataAssimilationBase initialization."""
    # This is an abstract base class, so we'll test it through inheritance
    class MockDA(DataAssimilationBase):
        def forward(self, state_in, observations, ensemble_members=None):
            return state_in
            
        def initialize_ensemble(self, background_state, num_members):
            return background_state
            
        def assimilate(self, ensemble, observations):
            return ensemble
    
    # Test initialization with default config
    mock_da = MockDA()
    assert mock_da.config == {}
    
    # Test initialization with custom config
    config = {"param": "value"}
    mock_da_config = MockDA(config)
    assert mock_da_config.config == config


def test_ensemble_generator_initialization():
    """Test EnsembleGenerator initialization."""
    # Test default initialization
    generator = EnsembleGenerator()
    assert generator.noise_std == 0.1
    assert generator.method == "gaussian"
    
    # Test custom initialization
    generator_custom = EnsembleGenerator(noise_std=0.2, method="perturbation")
    assert generator_custom.noise_std == 0.2
    assert generator_custom.method == "perturbation"


def test_ensemble_generator_tensor_forward():
    """Test EnsembleGenerator with tensor inputs."""
    generator = EnsembleGenerator(noise_std=0.1, method="gaussian")
    
    # Create input state
    batch_size = 2
    state_features = 8
    state = torch.randn(batch_size, state_features)
    num_members = 5
    
    # Generate ensemble
    ensemble = generator(state, num_members)
    
    # Check ensemble shape
    assert ensemble.shape == (batch_size, num_members, state_features)
    
    # Check that ensemble members are similar but not identical to original
    ensemble_mean = torch.mean(ensemble, dim=1)
    # Mean should be close to original state (within noise range)
    assert torch.allclose(ensemble_mean, state, atol=0.2)


def test_ensemble_generator_tensor_dropout_method():
    """Test EnsembleGenerator with dropout method."""
    generator = EnsembleGenerator(noise_std=0.1, method="dropout")
    
    # Create input state
    batch_size = 2
    state_features = 8
    state = torch.randn(batch_size, state_features)
    num_members = 4
    
    # Generate ensemble
    ensemble = generator(state, num_members)
    
    # Check ensemble shape
    assert ensemble.shape == (batch_size, num_members, state_features)


def test_ensemble_generator_tensor_perturbation_method():
    """Test EnsembleGenerator with perturbation method."""
    generator = EnsembleGenerator(noise_std=0.1, method="perturbation")
    
    # Create input state
    batch_size = 2
    state_features = 8
    state = torch.randn(batch_size, state_features)
    num_members = 4
    
    # Generate ensemble
    ensemble = generator(state, num_members)
    
    # Check ensemble shape
    assert ensemble.shape == (batch_size, num_members, state_features)


def test_ensemble_generator_graph_forward():
    """Test EnsembleGenerator with graph inputs."""
    generator = EnsembleGenerator(noise_std=0.1, method="gaussian")
    
    # Create graph input
    num_nodes = 4
    node_features = 6
    graph_state = Data(
        x=torch.randn(num_nodes, node_features),
        edge_index=torch.randint(0, num_nodes, (2, 8))
    )
    num_members = 3
    
    # Generate ensemble
    ensemble = generator(graph_state, num_members)
    
    # Check ensemble has correct structure
    assert hasattr(ensemble, 'x')
    # For graphs, ensemble creates [num_nodes, num_members, features] shape for x
    assert ensemble.x.shape == (num_nodes, num_members, node_features)
    # Edge index should remain the same
    assert torch.equal(ensemble.edge_index, graph_state.edge_index)


def test_ensemble_generator_graph_multiple_methods():
    """Test EnsembleGenerator with graph inputs using different methods."""
    methods = ["gaussian", "perturbation"]
    
    for method in methods:
        generator = EnsembleGenerator(noise_std=0.1, method=method)
        
        # Create graph input
        num_nodes = 3
        node_features = 5
        graph_state = Data(
            x=torch.randn(num_nodes, node_features),
            edge_index=torch.randint(0, num_nodes, (2, 6))
        )
        num_members = 2
        
        # Generate ensemble
        ensemble = generator(graph_state, num_members)
        
        # Check ensemble has correct structure
        assert hasattr(ensemble, 'x')
        assert ensemble.x.shape == (num_nodes, num_members, node_features)


def test_ensemble_generator_error_handling():
    """Test EnsembleGenerator error handling for invalid inputs."""
    generator = EnsembleGenerator()
    
    # Test invalid state type
    with pytest.raises(TypeError):
        generator("invalid_type", 5)


def test_ensemble_generator_consistency():
    """Test that ensemble generation is consistent for the same input."""
    generator = EnsembleGenerator(noise_std=0.01, method="gaussian")  # Small noise for consistency check
    
    # Create input state
    state = torch.ones(2, 4)  # Use ones for predictable results
    num_members = 3
    
    # Generate ensemble multiple times
    ensemble1 = generator(state, num_members)
    ensemble2 = generator(state, num_members)
    
    # Both should have same shape
    assert ensemble1.shape == ensemble2.shape
    
    # Members within each ensemble should be similar (since we used small noise)
    ensemble1_mean = torch.mean(ensemble1, dim=1)
    assert torch.allclose(ensemble1_mean, state, atol=0.1)


def test_ensemble_diversity():
    """Test that ensemble members are diverse (not identical)."""
    generator = EnsembleGenerator(noise_std=0.1, method="gaussian")
    
    # Create input state
    state = torch.ones(2, 4)  # Use ones
    num_members = 5
    
    # Generate ensemble
    ensemble = generator(state, num_members)
    
    # Check that ensemble members are not all identical
    # Calculate variance across ensemble dimension
    var_across_ensemble = torch.var(ensemble, dim=1, unbiased=False)
    # There should be some variance due to added noise
    assert not torch.allclose(var_across_ensemble, torch.zeros_like(var_across_ensemble), atol=1e-6)


if __name__ == "__main__":
    print("Running Data Assimilation Base tests...")
    
    test_data_assimilation_base_initialization()
    print("✓ Base initialization test passed")
    
    test_ensemble_generator_initialization()
    print("✓ Generator initialization test passed")
    
    test_ensemble_generator_tensor_forward()
    print("✓ Tensor forward test passed")
    
    test_ensemble_generator_tensor_dropout_method()
    print("✓ Tensor dropout method test passed")
    
    test_ensemble_generator_tensor_perturbation_method()
    print("✓ Tensor perturbation method test passed")
    
    test_ensemble_generator_graph_forward()
    print("✓ Graph forward test passed")
    
    test_ensemble_generator_graph_multiple_methods()
    print("✓ Graph multiple methods test passed")
    
    test_ensemble_generator_error_handling()
    print("✓ Error handling test passed")
    
    test_ensemble_generator_consistency()
    print("✓ Consistency test passed")
    
    test_ensemble_diversity()
    print("✓ Diversity test passed")
    
    print("\n✅ All Data Assimilation Base tests passed!")