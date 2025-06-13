import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from wronai.models.base import WronAIModel, ModelConfig
from wronai.models.mistral import WronAIMistral
from wronai.models.llama import WronAILlama
from wronai.models import load_model


class TestModelConfig:
    """Test ModelConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()

        assert config.model_name == "mistralai/Mistral-7B-v0.1"
        assert config.model_type == "causal_lm"
        assert config.quantization_enabled == True
        assert config.lora_enabled == True
        assert config.max_sequence_length == 2048
        assert config.torch_dtype == "bfloat16"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            model_name="test-model",
            max_sequence_length=1024,
            lora_r=32
        )

        assert config.model_name == "test-model"
        assert config.max_sequence_length == 1024
        assert config.lora_r == 32

    def test_post_init(self):
        """Test post-initialization setup."""
        config = ModelConfig()

        # Check default Polish tokens are set
        assert "<polish>" in config.polish_tokens
        assert "</polish>" in config.polish_tokens
        assert "<question>" in config.polish_tokens
        assert "<answer>" in config.polish_tokens

        # Check default LoRA target modules
        assert "q_proj" in config.lora_target_modules
        assert "v_proj" in config.lora_target_modules


class TestWronAIModel:
    """Test base WronAIModel class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return ModelConfig(
            model_name="microsoft/DialoGPT-small",
            quantization_enabled=False,  # Disable for testing
            lora_enabled=False  # Disable for testing
        )

    @pytest.fixture
    def mock_model(self, mock_config):
        """Create mock model for testing."""
        # Create a concrete implementation for testing
        class TestWronAIModel(WronAIModel):
            def preprocess_text(self, text: str) -> str:
                return text.strip()

            def postprocess_text(self, text: str) -> str:
                return text.strip()

        return TestWronAIModel(mock_config)

    def test_model_initialization(self, mock_model):
        """Test model initialization."""
        assert mock_model.config.model_name == "microsoft/DialoGPT-small"
        assert mock_model.model is None
        assert mock_model.tokenizer is None
        assert mock_model._is_quantized == False
        assert mock_model._has_lora == False

    @patch('wronai.models.base.AutoModelForCausalLM')
    @patch('wronai.models.base.AutoTokenizer')
    def test_load_model_and_tokenizer(self, mock_tokenizer_cls, mock_model_cls, mock_model):
        """Test loading model and tokenizer."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.add_tokens.return_value = 5
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_transformer_model = Mock()
        mock_transformer_model.resize_token_embeddings = Mock()
        mock_model_cls.from_pretrained.return_value = mock_transformer_model

        # Load model and tokenizer
        mock_model.load_model()
        mock_model.load_tokenizer()

        # Verify calls
        mock_model_cls.from_pretrained.assert_called_once()
        mock_tokenizer_cls.from_pretrained.assert_called_once()

        # Verify model state
        assert mock_model.model == mock_transformer_model
        assert mock_model.tokenizer == mock_tokenizer
        assert mock_tokenizer.pad_token == "<eos>"

    def test_parameter_counting(self, mock_model):
        """Test parameter counting methods."""
        # Mock model with parameters
        param1 = torch.randn(100, 200)  # 20,000 parameters
        param2 = torch.randn(50, 50)    # 2,500 parameters

        mock_model.model = Mock()
        mock_model.model.parameters.return_value = [param1, param2]

        total_params = mock_model.get_parameter_count()
        assert total_params == 22500

    def test_trainable_parameter_counting(self, mock_model):
        """Test trainable parameter counting."""
        # Mock parameters with different requires_grad
        param1 = torch.randn(100, 200)
        param1.requires_grad = True

        param2 = torch.randn(50, 50)
        param2.requires_grad = False

        mock_model.model = Mock()
        mock_model.model.parameters.return_value = [param1, param2]

        trainable_params = mock_model.get_trainable_parameter_count()
        assert trainable_params == 20000  # Only param1

    @patch('wronai.models.base.torch.cuda.is_available')
    @patch('wronai.models.base.torch.cuda.memory_allocated')
    @patch('wronai.models.base.torch.cuda.get_device_properties')
    def test_memory_usage(self, mock_props, mock_allocated, mock_cuda_available, mock_model):
        """Test memory usage calculation."""
        mock_cuda_available.return_value = True
        mock_allocated.return_value = 4 * 1024**3  # 4GB

        mock_device_props = Mock()
        mock_device_props.total_memory = 8 * 1024**3  # 8GB
        mock_props.return_value = mock_device_props

        memory_info = mock_model.get_memory_usage()

        assert memory_info["gpu_memory_used"] == 4.0
        assert memory_info["gpu_memory_total"] == 8.0
        assert memory_info["gpu_memory_percent"] == 50.0

    def test_memory_usage_no_cuda(self, mock_model):
        """Test memory usage when CUDA is not available."""
        with patch('wronai.models.base.torch.cuda.is_available', return_value=False):
            memory_info = mock_model.get_memory_usage()

            assert memory_info["gpu_memory_used"] == 0
            assert memory_info["gpu_memory_total"] == 0


class TestWronAIMistral:
    """Test WronAIMistral implementation."""

    @pytest.fixture
    def mistral_config(self):
        """Create Mistral configuration for testing."""
        return ModelConfig(
            model_name="microsoft/DialoGPT-small",  # Use small model for testing
            quantization_enabled=False,
            lora_enabled=False
        )

    @pytest.fixture
    def mistral_model(self, mistral_config):
        """Create Mistral model for testing."""
        return WronAIMistral(mistral_config)

    def test_mistral_initialization(self, mistral_model):
        """Test Mistral model initialization."""
        assert mistral_model.model_type == "mistral"
        assert isinstance(mistral_model.config, ModelConfig)

    def test_polish_text_preprocessing(self, mistral_model):
        """Test Polish text preprocessing."""
        input_text = "  Witaj   świecie!  To jest   test.  "
        processed = mistral_model.preprocess_text(input_text)

        # Should normalize whitespace and fix typography
        assert "   " not in processed
        assert processed.strip() == processed
        assert len(processed) > 0

    def test_polish_typography_fixing(self, mistral_model):
        """Test Polish typography fixes."""
        # Test quote standardization
        text_with_quotes = '"Witaj świecie" powiedział.'
        processed = mistral_model._fix_polish_typography(text_with_quotes)
        assert '"' in processed or '„' in processed

        # Test dash replacement
        text_with_dashes = "Test - to jest myślnik"
        processed = mistral_model._fix_polish_typography(text_with_dashes)
        assert " – " in processed

    def test_text_postprocessing(self, mistral_model):
        """Test text postprocessing."""
        # Text with special tokens
        generated_text = "<polish>Witaj świecie!</polish> To jest test."
        processed = mistral_model.postprocess_text(generated_text)

        # Should remove special tokens
        assert "<polish>" not in processed
        assert "</polish>" not in processed
        assert "Witaj świecie!" in processed

    def test_sentence_structure_fixing(self, mistral_model):
        """Test sentence structure fixing."""
        text = "witaj świecie. jak się masz"
        fixed = mistral_model._fix_sentence_structure(text)

        # Should capitalize properly
        assert fixed.startswith("Witaj")
        assert ". Jak" in fixed or fixed.endswith(".")

    def test_capitalization_fixing(self, mistral_model):
        """Test Polish capitalization."""
        text = "witaj. jak się masz? dobrze."
        fixed = mistral_model._fix_capitalization(text)

        assert fixed.startswith("Witaj")
        assert ". Jak" in fixed
        assert "? Dobrze" in fixed

    @patch('wronai.models.mistral.MistralForCausalLM')
    def test_model_loading(self, mock_mistral_cls, mistral_model):
        """Test Mistral model loading."""
        mock_model = Mock()
        mock_mistral_cls.from_pretrained.return_value = mock_model

        mistral_model.load_model()

        mock_mistral_cls.from_pretrained.assert_called_once()
        assert mistral_model.model == mock_model

    def test_model_info(self, mistral_model):
        """Test getting model information."""
        # Mock model for info
        mistral_model.model = Mock()
        mistral_model.model.parameters.return_value = [torch.randn(100, 100)]

        info = mistral_model.get_model_info()

        assert info["model_type"] == "mistral"
        assert "total_parameters" in info
        assert "config" in info

    def test_memory_estimation(self, mistral_model):
        """Test memory usage estimation."""
        # Mock parameter count
        mistral_model.get_parameter_count = Mock(return_value=7000000000)  # 7B parameters

        estimation = mistral_model.estimate_memory_usage(batch_size=1)

        assert "model_memory" in estimation
        assert "activation_memory" in estimation
        assert "total_estimated" in estimation
        assert estimation["total_estimated"] > 0


class TestWronAILlama:
    """Test WronAILlama implementation."""

    @pytest.fixture
    def llama_config(self):
        """Create LLaMA configuration for testing."""
        return ModelConfig(
            model_name="microsoft/DialoGPT-small",  # Use small model for testing
            quantization_enabled=False,
            lora_enabled=False
        )

    @pytest.fixture
    def llama_model(self, llama_config):
        """Create LLaMA model for testing."""
        return WronAILlama(llama_config)

    def test_llama_initialization(self, llama_model):
        """Test LLaMA model initialization."""
        assert llama_model.model_type == "llama"

    def test_llama_prompt_formatting(self, llama_model):
        """Test LLaMA-specific prompt formatting."""
        prompt = "Test prompt"
        formatted = llama_model._format_for_llama(prompt)

        # Should add Alpaca-style formatting
        assert "### Instrukcja:" in formatted or "###" in formatted
        assert "### Odpowiedź:" in formatted or prompt in formatted

    def test_llama_output_cleaning(self, llama_model):
        """Test LLaMA output cleaning."""
        output_with_artifacts = "### Odpowiedź:\nTo jest odpowiedź. ### Koniec"
        cleaned = llama_model._clean_llama_output(output_with_artifacts)

        # Should remove formatting artifacts
        assert "### Odpowiedź:" not in cleaned
        assert "To jest odpowiedź" in cleaned

    def test_chat_with_history(self, llama_model):
        """Test chat with conversation history."""
        # Mock tokenizer and model for generation
        llama_model.tokenizer = Mock()
        llama_model.tokenizer.return_value = Mock()
        llama_model.tokenizer.return_value.to.return_value = Mock()
        llama_model.tokenizer.return_value.input_ids = torch.tensor([[1, 2, 3]])
        llama_model.tokenizer.decode.return_value = "Test response"

        llama_model.model = Mock()
        llama_model.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        llama_model.device = "cpu"

        # Mock postprocess_text
        llama_model.postprocess_text = Mock(return_value="Clean response")

        history = [
            {"user": "Jak się masz?", "assistant": "Dobrze, dziękuję!"}
        ]

        response = llama_model.chat_with_history("Co robisz?", history)

        assert isinstance(response, str)
        llama_model.model.generate.assert_called_once()


class TestModelFactory:
    """Test model factory functions."""

    @patch('wronai.models.WronAIMistral')
    def test_load_mistral_model(self, mock_mistral_cls):
        """Test loading Mistral model through factory."""
        mock_model = Mock()
        mock_mistral_cls.from_pretrained.return_value = mock_model

        model = load_model("mistralai/Mistral-7B-v0.1")

        mock_mistral_cls.from_pretrained.assert_called_once()

    @patch('wronai.models.WronAILlama')
    def test_load_llama_model(self, mock_llama_cls):
        """Test loading LLaMA model through factory."""
        mock_model = Mock()
        mock_llama_cls.from_pretrained.return_value = mock_model

        model = load_model("llama-model-name")

        mock_llama_cls.from_pretrained.assert_called_once()

    @patch('wronai.models.load_quantized_model')
    @patch('wronai.models.WronAIMistral')
    def test_load_model_with_quantization(self, mock_mistral_cls, mock_quantize):
        """Test loading model with quantization."""
        mock_model = Mock()
        mock_mistral_cls.from_pretrained.return_value = mock_model
        mock_quantize.return_value = mock_model

        model = load_model("mistralai/Mistral-7B-v0.1", quantize=True)

        mock_quantize.assert_called_once_with(mock_model)


class TestModelIntegration:
    """Integration tests for models."""

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_model_loading_integration(self, test_model_config):
        """Test actual model loading (requires GPU and network)."""
        pytest.skip("Requires actual model download and GPU")

        # This would test actual model loading
        # model = WronAIMistral(test_model_config)
        # model.load_model()
        # model.load_tokenizer()
        # assert model.model is not None
        # assert model.tokenizer is not None

    @pytest.mark.polish
    def test_polish_text_quality(self):
        """Test Polish text processing quality."""
        # Test with various Polish texts
        test_texts = [
            "Witaj świecie! Jak się masz?",
            "To jest test polskiego tekstu z diakrytykami: ąćęłńóśźż",
            "„Cytat w polskich cudzysłowach" - powiedział ktoś.",
            "Test z myślnikami - pierwszy - drugi - trzeci.",
            "Wielokropek... i inne znaki."
        ]

        config = ModelConfig(quantization_enabled=False, lora_enabled=False)
        model = WronAIMistral(config)

        for text in test_texts:
            processed = model.preprocess_text(text)

            # Basic quality checks
            assert len(processed) > 0
            assert processed == processed.strip()
            # Should preserve Polish characters
            polish_chars = set('ąćęłńóśźż')
            if any(char in text.lower() for char in polish_chars):
                assert any(char in processed.lower() for char in polish_chars)


if __name__ == "__main__":
    pytest.main([__file__])