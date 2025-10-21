import os
import tempfile
import logging
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import requests
from openai import OpenAI
import hashlib
import re
import numpy as np

logger = logging.getLogger(__name__)

class ThumbnailGenerator:
    """Generates YouTube-ready thumbnails from text prompts."""
    
    def __init__(self):
        """Initialize the thumbnail generator."""
        # Azure OpenAI configuration
        self.azure_endpoint_dalle = os.getenv("AZURE_OPENAI_ENDPOINT_DALLE")
        self.azure_api_key = os.getenv('AZURE_API_KEY_DALLE')
        self.azure_deployment = os.getenv('AZURE_DEPLOYMENT_NAME_DALLE')
        
        # GPT endpoint for caption generation (different endpoint and API key)
        self.gpt_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT_GPT')
        self.gpt_api_key = os.getenv('AZURE_API_KEY_GPT')
        self.gpt_deployment = os.getenv('AZURE_DEPLOYMENT_NAME_DALLE')
        
        # Check if Azure API key is available
        if self.azure_api_key:
            self.use_azure = True
            logger.info("Azure OpenAI configured")
        else:
            self.use_azure = False
            logger.info("No Azure API key found, using gradient backgrounds only")
            
        self.thumbnail_size = (1280, 720)  # YouTube thumbnail size
        self.temp_dir = os.path.join(tempfile.gettempdir(), 'thumbnails')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Color schemes for different types of content
        self.color_schemes = {
            'quiz': {
                'bg': '#FF6B6B',
                'text': '#FFFFFF',
                'accent': '#FFE66D'
            },
            'riddle': {
                'bg': '#4ECDC4',
                'text': '#FFFFFF',
                'accent': '#45B7D1'
            },
            'challenge': {
                'bg': '#A8E6CF',
                'text': '#2C3E50',
                'accent': '#FF8B94'
            },
            'default': {
                'bg': '#667EEA',
                'text': '#FFFFFF',
                'accent': '#764BA2'
            }
        }
    
    def generate_thumbnail(self, prompt: str) -> Optional[str]:
        """
        Generate a thumbnail from the given prompt.
        
        Args:
            prompt: Text prompt for the thumbnail
            
        Returns:
            Path to the generated thumbnail file or None if failed
        """
        try:
            # Generate catchy caption for long prompts
            
            display_text, gpt_success = self._generate_catchy_caption(prompt)
            content_type = self._analyze_content_type(display_text)
            
            # Generate background image
            if not gpt_success:
                print("🎨 GPT failed, using gradient background fallback")
                logger.info("GPT failed, using gradient background fallback")
                background_path = self._generate_gradient_background(content_type)
            else:
                background_path = self._generate_background(prompt, content_type)
            
            if not background_path:
                logger.error("Failed to generate background image")
                return None
            
            # Create the final thumbnail with text overlay
            thumbnail_path = self._create_thumbnail_with_text(
                background_path, display_text, content_type
            )
            
            # Clean up temporary background file
            if os.path.exists(background_path):
                os.remove(background_path)
            
            return thumbnail_path
            
        except Exception as e:
            logger.error(f"Error generating thumbnail: {str(e)}")
            return None
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and format the prompt text."""
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', prompt.strip())
        
        # Capitalize first letter of each word for better display
        cleaned = cleaned.title()
        
        return cleaned
    
    def _generate_catchy_caption(self, prompt: str) -> tuple[str, bool]:
        """Generate a catchy, short caption using GPT for long prompts.
        
        Returns:
            tuple: (caption_text, gpt_success)
        """
        try:
            # Only use GPT for prompts longer than 50 characters
            if len(prompt) <= 50:
                return self._clean_prompt(prompt), True
            
            # Check if GPT API is available
            if not self.gpt_api_key or not self.gpt_endpoint:
                print("🔄 No GPT API key/endpoint available")
                logger.info("No GPT API key/endpoint available")
                return self._clean_prompt(prompt), False
            
            print(f"🤖 Generating catchy caption for long prompt...")
            logger.info(f"🤖 Generating catchy caption for long prompt: {prompt}")
            
            # GPT API call
            url = f"{self.gpt_endpoint}/openai/deployments/{self.gpt_deployment}/chat/completions?api-version=2025-01-01-preview"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.gpt_api_key}"
            }
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Create a catchy, short caption (max 6 words) for a YouTube thumbnail about: {prompt}. Make it engaging and clickable. Only return the caption, nothing else."
                    }
                ],
                "max_completion_tokens": 50,
                "temperature": 1,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "model": self.gpt_deployment
            }
            
            print(f"📤 Sending GPT request to: {url}")
            logger.info(f"📤 Sending GPT request to: {url}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            print(f"📊 GPT Response status: {response.status_code}")
            logger.info(f"📊 GPT Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Got GPT response: {result}")
                logger.info(f"✅ Got GPT response: {result}")
                
                if 'choices' in result and len(result['choices']) > 0:
                    caption = result['choices'][0]['message']['content'].strip()
                    print(f"🎯 Generated caption: {caption}")
                    logger.info(f"🎯 Generated caption: {caption}")
                    return caption, True
                else:
                    print("❌ No caption in GPT response")
                    logger.error("❌ No caption in GPT response")
                    return self._clean_prompt(prompt), False
            else:
                print(f"❌ GPT failed: {response.status_code}")
                print(f"Response: {response.text}")
                logger.error(f"❌ GPT failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return self._clean_prompt(prompt), False
                
        except Exception as e:
            print(f"❌ GPT caption generation failed: {str(e)}")
            logger.warning(f"GPT caption generation failed: {str(e)}")
            return self._clean_prompt(prompt), False
    
    def _validate_image_quality(self, img: Image.Image) -> bool:
        """Validate image quality to prevent artifacts and distortions."""
        try:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get image statistics
            width, height = img.size
            
            # Check minimum dimensions
            if width < 100 or height < 100:
                print("❌ Image too small for quality validation")
                return False
            
            # Check for completely black or white images (likely artifacts)
            img_array = np.array(img)
            mean_brightness = np.mean(img_array)
            
            if mean_brightness < 10:  # Too dark
                print("❌ Image too dark (likely artifact)")
                return False
            elif mean_brightness > 245:  # Too bright
                print("❌ Image too bright (likely artifact)")
                return False
            
            # Check for uniform color (likely gradient, which is good)
            std_dev = np.std(img_array)
            if std_dev < 5:  # Very uniform color
                print("✅ Uniform color detected (good for backgrounds)")
                return True
            
            # Check for reasonable color distribution
            if std_dev > 100:  # Very high variation (might have artifacts)
                print("⚠️ High color variation detected, checking for artifacts")
                
                # Check for extreme pixel values that might indicate artifacts
                max_pixel = np.max(img_array)
                min_pixel = np.min(img_array)
                
                if max_pixel - min_pixel > 200:  # Very high contrast
                    print("⚠️ Very high contrast detected")
                    # Additional check for noise patterns
                    if self._check_for_noise_patterns(img_array):
                        print("❌ Noise patterns detected (likely artifacts)")
                        return False
            
            # Check for reasonable file size (very small files might be corrupted)
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            file_size = len(buffer.getvalue())
            
            if file_size < 1000:  # Less than 1KB is suspicious
                print("❌ File size too small (likely corrupted)")
                return False
            
            print("✅ Image quality validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Image quality validation error: {str(e)}")
            logger.warning(f"Image quality validation error: {str(e)}")
            return False
    
    def _check_for_noise_patterns(self, img_array: np.ndarray) -> bool:
        """Check for noise patterns that might indicate artifacts."""
        try:
            # Check for salt-and-pepper noise
            # Look for isolated extreme pixels
            height, width = img_array.shape[:2]
            
            # Sample a few regions to check for noise
            sample_regions = [
                img_array[0:50, 0:50],      # Top-left
                img_array[0:50, -50:],      # Top-right
                img_array[-50:, 0:50],      # Bottom-left
                img_array[-50:, -50:],      # Bottom-right
                img_array[height//2-25:height//2+25, width//2-25:width//2+25]  # Center
            ]
            
            for region in sample_regions:
                region_std = np.std(region)
                region_mean = np.mean(region)
                
                # Check for extreme outliers
                outliers = np.abs(region - region_mean) > (3 * region_std)
                outlier_ratio = np.sum(outliers) / outliers.size
                
                if outlier_ratio > 0.1:  # More than 10% outliers
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Noise pattern check error: {str(e)}")
            return False
    
    def _detect_objects_and_find_text_region(self, img: Image.Image) -> tuple[int, int]:
        """Detect objects (like hands) and find the best region for text placement."""
        try:
            print("🔍 Analyzing image for object detection...")
            logger.info("Analyzing image for object detection")
            
            # Convert to numpy array for analysis
            img_array = np.array(img)
            height, width = img_array.shape[:2]
            
            # Define regions to check (avoiding center where objects might be)
            regions = [
                # Top regions
                {"name": "top-left", "coords": (0, 0, width//3, height//3), "priority": 3},
                {"name": "top-center", "coords": (width//3, 0, 2*width//3, height//3), "priority": 2},
                {"name": "top-right", "coords": (2*width//3, 0, width, height//3), "priority": 3},
                
                # Middle regions (avoid center)
                {"name": "middle-left", "coords": (0, height//3, width//3, 2*height//3), "priority": 4},
                {"name": "middle-right", "coords": (2*width//3, height//3, width, 2*height//3), "priority": 4},
                
                # Bottom regions
                {"name": "bottom-left", "coords": (0, 2*height//3, width//3, height), "priority": 3},
                {"name": "bottom-center", "coords": (width//3, 2*height//3, 2*width//3, height), "priority": 2},
                {"name": "bottom-right", "coords": (2*width//3, 2*height//3, width, height), "priority": 3},
            ]
            
            # Analyze each region for object detection
            region_scores = []
            
            for region in regions:
                x1, y1, x2, y2 = region["coords"]
                region_data = img_array[y1:y2, x1:x2]
                
                # Calculate region statistics
                mean_brightness = np.mean(region_data)
                std_dev = np.std(region_data)
                color_variance = np.var(region_data, axis=(0,1)) if len(region_data.shape) == 3 else np.var(region_data)
                
                # Score based on suitability for text placement
                score = 0
                
                # Prefer regions with moderate brightness (not too dark/light)
                if 50 < mean_brightness < 200:
                    score += 2
                elif 30 < mean_brightness < 220:
                    score += 1
                
                # Prefer regions with moderate variation (not too uniform or chaotic)
                if 20 < std_dev < 80:
                    score += 2
                elif 10 < std_dev < 100:
                    score += 1
                
                # Prefer regions with lower color variance (more uniform)
                if isinstance(color_variance, np.ndarray):
                    color_variance = np.mean(color_variance)
                if color_variance < 1000:
                    score += 1
                
                # Add priority bonus
                score += region["priority"]
                
                # Check for potential objects (high contrast areas)
                if std_dev > 60:
                    score -= 1  # Penalize high-contrast areas (might have objects)
                
                region_scores.append({
                    "name": region["name"],
                    "coords": region["coords"],
                    "score": score,
                    "mean_brightness": mean_brightness,
                    "std_dev": std_dev
                })
            
            # Sort by score (highest first)
            region_scores.sort(key=lambda x: x["score"], reverse=True)
            
            # Get the best region
            best_region = region_scores[0]
            x1, y1, x2, y2 = best_region["coords"]
            
            # Calculate center of the best region
            text_x = (x1 + x2) // 2
            text_y = (y1 + y2) // 2
            
            print(f"✅ Best text region: {best_region['name']} (score: {best_region['score']})")
            print(f"📍 Text position: ({text_x}, {text_y})")
            logger.info(f"Best text region: {best_region['name']} (score: {best_region['score']})")
            logger.info(f"Text position: ({text_x}, {text_y})")
            
            return text_x, text_y
            
        except Exception as e:
            print(f"❌ Object detection error: {str(e)}")
            logger.warning(f"Object detection error: {str(e)}")
            # Fallback to center
            return img.size[0] // 2, img.size[1] // 2
    
    def _get_adaptive_text_colors(self, img: Image.Image, text_x: int, text_y: int, text_height: int) -> tuple[tuple, float]:
        """Analyze background brightness around text area to determine optimal text colors."""
        try:
            print("🎨 Analyzing background brightness for adaptive text colors...")
            logger.info("Analyzing background brightness for adaptive text colors")
            
            # Convert to numpy array for analysis
            img_array = np.array(img)
            height, width = img_array.shape[:2]
            
            # Define text area for analysis (wider than text to get background context)
            text_area_width = min(400, width // 2)  # Analyze wider area
            text_area_height = min(text_height + 40, height // 3)  # Analyze taller area
            
            # Calculate analysis bounds
            x1 = max(0, text_x - text_area_width // 2)
            x2 = min(width, text_x + text_area_width // 2)
            y1 = max(0, text_y - 20)
            y2 = min(height, text_y + text_area_height)
            
            # Extract the text area
            text_area = img_array[y1:y2, x1:x2]
            
            # Calculate brightness statistics
            if len(text_area.shape) == 3:  # RGB image
                # Convert to grayscale for brightness analysis
                gray_area = np.mean(text_area, axis=2)
            else:  # Already grayscale
                gray_area = text_area
            
            # Calculate brightness metrics
            mean_brightness = np.mean(gray_area)
            median_brightness = np.median(gray_area)
            std_brightness = np.std(gray_area)
            
            print(f"📊 Background analysis: mean={mean_brightness:.1f}, median={median_brightness:.1f}, std={std_brightness:.1f}")
            logger.info(f"Background analysis: mean={mean_brightness:.1f}, median={median_brightness:.1f}, std={std_brightness:.1f}")
            
            # Determine if background is light or dark
            # Use median for more robust detection (less affected by outliers)
            is_light_background = median_brightness > 128
            
            # Calculate contrast ratio
            contrast_ratio = std_brightness / (mean_brightness + 1)  # Avoid division by zero
            
            # Choose bold text color based on background (no shadows)
            if is_light_background:
                # Light background -> dark text
                if mean_brightness > 200:
                    # Very light background
                    text_color = (0, 0, 0)  # Pure black for maximum contrast
                else:
                    # Medium light background
                    text_color = (10, 10, 10)  # Very dark text
            else:
                # Dark background -> light text
                if mean_brightness < 50:
                    # Very dark background
                    text_color = (255, 255, 255)  # Pure white text
                else:
                    # Medium dark background
                    text_color = (255, 255, 255)  # Pure white for maximum visibility
            
            # Always use maximum contrast for thumbnails
            if is_light_background:
                text_color = (0, 0, 0)  # Pure black for light backgrounds
            else:
                text_color = (255, 255, 255)  # Pure white for dark backgrounds
            
            print(f"✅ Adaptive colors: background={'light' if is_light_background else 'dark'}, contrast={contrast_ratio:.2f}")
            logger.info(f"Adaptive colors: background={'light' if is_light_background else 'dark'}, contrast={contrast_ratio:.2f}")
            
            return text_color, contrast_ratio
            
        except Exception as e:
            print(f"❌ Adaptive color analysis error: {str(e)}")
            logger.warning(f"Adaptive color analysis error: {str(e)}")
            # Fallback to default colors
            return (255, 255, 255), 0.5
    
    def _analyze_content_type(self, prompt: str) -> str:
        """Analyze the prompt to determine content type for styling."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['quiz', 'guess', 'test']):
            return 'quiz'
        elif any(word in prompt_lower for word in ['riddle', 'puzzle', 'solve']):
            return 'riddle'
        elif any(word in prompt_lower for word in ['challenge', 'only', 'can you']):
            return 'challenge'
        else:
            return 'default'
    
    def _generate_background(self, prompt: str, content_type: str) -> Optional[str]:
        """Generate background image using Azure OpenAI or fallback to gradient."""
        # Try Azure OpenAI first if available
        
        if self.use_azure:
            logger.info("Use azure called")
            dalle_result = self._generate_dalle_background(prompt, content_type)
            if dalle_result:
                return dalle_result
            else:
                logger.warning("Azure OpenAI failed, falling back to gradient background")
        
        # Fallback to gradient background
        logger.info("🎨 Falling back to gradient background")
        return self._generate_gradient_background(content_type)
    
    def _generate_dalle_background(self, prompt: str, content_type: str) -> Optional[str]:
        """Generate background using Azure OpenAI DALL-E with direct REST API."""
        try:
            print(f"🔄 Calling Azure OpenAI with prompt: {prompt}")
            print(f"🔗 Endpoint: {self.azure_endpoint_dalle}")
            print(f"🔑 API Key: {'Set' if self.azure_api_key else 'Not set'}")
            logger.info(f"🔄 Calling Azure OpenAI with prompt: {prompt}")
            logger.info(f"🔗 Endpoint: {self.azure_endpoint_dalle}")
            logger.info(f"🔑 API Key: {'Set' if self.azure_api_key else 'Not set'}")
            
            # Create a prompt for realistic background generation (content-filter safe)
            bg_prompt = f"Create a beautiful landscape background for a YouTube thumbnail about: {prompt}. " \
                       f"Style: natural lighting, soft textures, muted colors, clean composition, " \
                       f"no text, no people, no faces, no hands, suitable for overlay text. " \
                       f"Format: landscape, 16:9 aspect ratio. " \
                       f"Requirements: high quality, peaceful environment, natural setting, " \
                       f"gentle lighting, outdoor scene, professional design."
            
            # Direct REST API call (exactly like your curl command)
            url = f"{self.azure_endpoint_dalle}/openai/deployments/{self.azure_deployment}/images/generations?api-version=2024-02-01"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.azure_api_key}"
            }
            
            payload = {
                "model": "dall-e-3",
                "prompt": bg_prompt,
                "size": "1024x1024",
                "style": "natural",
                "quality": "hd",
                "n": 1
            }
            
            print(f"📤 Sending direct REST request to: {url}")
            print(f"📋 Payload: {payload}")
            logger.info(f"📤 Sending direct REST request to: {url}")
            logger.info(f"📋 Payload: {payload}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            print(f"📊 Response status: {response.status_code}")
            logger.info(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Got response from Azure OpenAI!")
                print(f"📄 Full response: {result}")
                logger.info(f"✅ Got response from Azure OpenAI!")
                logger.info(f"📄 Full response: {result}")
                
                if 'data' in result and len(result['data']) > 0:
                    image_url = result['data'][0]['url']
                    print(f"🖼️  Image URL: {image_url}")
                    logger.info(f"🖼️  Image URL: {image_url}")
                    
                    # Download the image
                    print(f"📥 Downloading image...")
                    logger.info(f"📥 Downloading image...")
                    img_response = requests.get(image_url, timeout=30)
                    img_response.raise_for_status()
                    
                    # Save to temporary file
                    temp_path = os.path.join(self.temp_dir, f"bg_{hashlib.md5(prompt.encode()).hexdigest()}.png")
                    
                    with open(temp_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    print(f"💾 Saved image to: {temp_path}")
                    logger.info(f"💾 Saved image to: {temp_path}")
                    
                    # Resize to thumbnail dimensions
                    img = Image.open(temp_path)
                    img = img.resize(self.thumbnail_size, Image.Resampling.LANCZOS)
                    
                    # Quality validation - check for artifacts and distortions
                    if self._validate_image_quality(img):
                        img.save(temp_path, 'PNG', quality=95)
                        print("🎉 Azure OpenAI background generated successfully!")
                        logger.info("🎉 Azure OpenAI background generated successfully!")
                        return temp_path
                    else:
                        print("⚠️ Image quality validation failed, using gradient fallback")
                        logger.warning("Image quality validation failed, using gradient fallback")
                        return None
                else:
                    print("❌ No image data in response")
                    logger.error("❌ No image data in response")
                    return None
            else:
                print(f"❌ Azure OpenAI failed: {response.status_code}")
                print(f"Response: {response.text}")
                logger.error(f"❌ Azure OpenAI failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
            
        except Exception as e:
            print(f"❌ Azure OpenAI failed: {str(e)}")
            logger.warning(f"Azure OpenAI background generation failed: {str(e)}")
            return None
    
    def _generate_gradient_background(self, content_type: str) -> Optional[str]:
        """Generate a gradient background as fallback."""
        try:
            colors = self.color_schemes.get(content_type, self.color_schemes['default'])
            
            # Create gradient background
            img = Image.new('RGB', self.thumbnail_size, colors['bg'])
            draw = ImageDraw.Draw(img)
            
            # Create a simple gradient effect
            for y in range(self.thumbnail_size[1]):
                ratio = y / self.thumbnail_size[1]
                r = int(int(colors['bg'][1:3], 16) * (1 - ratio) + int(colors['accent'][1:3], 16) * ratio)
                g = int(int(colors['bg'][3:5], 16) * (1 - ratio) + int(colors['accent'][3:5], 16) * ratio)
                b = int(int(colors['bg'][5:7], 16) * (1 - ratio) + int(colors['accent'][5:7], 16) * ratio)
                
                draw.line([(0, y), (self.thumbnail_size[0], y)], fill=(r, g, b))
            
            # Add some visual elements
            self._add_background_elements(draw, colors)
            
            # Save to temporary file
            temp_path = os.path.join(self.temp_dir, f"gradient_{content_type}_{hashlib.md5(str(content_type).encode()).hexdigest()[:8]}.png")
            img.save(temp_path, 'PNG', quality=95)
            
            logger.info(f"Gradient background generated successfully for {content_type}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Gradient background generation failed: {str(e)}")
            return None
    
    def _add_background_elements(self, draw: ImageDraw.Draw, colors: dict):
        """Add visual elements to the background."""
        # Add some geometric shapes for visual interest
        width, height = self.thumbnail_size
        
        # Add circles
        for i in range(3):
            x = width // 4 + i * (width // 3)
            y = height // 4 + (i % 2) * (height // 2)
            size = 100 + i * 50
            draw.ellipse([x - size//2, y - size//2, x + size//2, y + size//2], 
                        fill=colors['accent'], outline=None)
    
    def _create_thumbnail_with_text(self, background_path: str, text: str, content_type: str) -> Optional[str]:
        """Create the final thumbnail with text overlay."""
        try:
            # Open background image
            img = Image.open(background_path)
            
            # No overlay needed - using smart positioning and adaptive colors
            
            # Add text
            draw = ImageDraw.Draw(img)
            colors = self.color_schemes.get(content_type, self.color_schemes['default'])
            
            # Try to load a bold font for thumbnail visibility
            try:
                # Try to use a bold font with larger size
                font_size = 96  # Increased from 72 to 96
                font = ImageFont.truetype("arialbd.ttf", font_size)  # Bold Arial
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", font_size)
                except:
                    try:
                        # Try regular Arial with larger size
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        try:
                            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
                        except:
                            # Fallback to default but larger
                            font = ImageFont.load_default()
                            font_size = 96
            
            # Split text into multiple lines if it's too long
            def split_text_into_lines(text, max_width, font, draw):
                """Split text into multiple lines that fit within max_width."""
                words = text.split()
                lines = []
                current_line = []
                
                for word in words:
                    # Test if adding this word would exceed max_width
                    test_line = ' '.join(current_line + [word])
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    line_width = bbox[2] - bbox[0]
                    
                    if line_width <= max_width:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                        else:
                            # Word is too long, add it anyway
                            lines.append(word)
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                return lines
            
            # Maximum width for text (80% of thumbnail width)
            max_text_width = int(self.thumbnail_size[0] * 0.8)
            lines = split_text_into_lines(text, max_text_width, font, draw)
            
            # Limit to maximum 3 lines
            if len(lines) > 3:
                lines = lines[:3]
                # Add "..." to the last line if it was truncated
                if len(lines) == 3:
                    last_line = lines[2]
                    bbox = draw.textbbox((0, 0), last_line + "...", font=font)
                    if (bbox[2] - bbox[0]) <= max_text_width:
                        lines[2] = last_line + "..."
            
            # Calculate total text height
            bbox = draw.textbbox((0, 0), 'A', font=font)
            line_height = (bbox[3] - bbox[1]) + 10  # Add some spacing between lines
            total_text_height = len(lines) * line_height
            
            # Use smart object detection to find best text position
            try:
                # Create a copy of the image for analysis
                analysis_img = img.copy()
                best_x, best_y = self._detect_objects_and_find_text_region(analysis_img)
                
                # Adjust for text block centering
                start_y = best_y - (total_text_height // 2)
                
                # Ensure text stays within image bounds
                if start_y < 20:
                    start_y = 20
                elif start_y + total_text_height > self.thumbnail_size[1] - 20:
                    start_y = self.thumbnail_size[1] - total_text_height - 20
                    
                print(f"🎯 Using smart text positioning: ({best_x}, {start_y})")
                logger.info(f"Using smart text positioning: ({best_x}, {start_y})")
                
            except Exception as e:
                print(f"⚠️ Smart positioning failed, using center: {str(e)}")
                logger.warning(f"Smart positioning failed, using center: {str(e)}")
                # Fallback to center
                start_y = (self.thumbnail_size[1] - total_text_height) // 2
                best_x = self.thumbnail_size[0] // 2
            
            # Analyze background brightness to determine text color
            try:
                text_color, contrast_ratio = self._get_adaptive_text_colors(img, best_x, start_y, total_text_height)
                print(f"🎨 Adaptive text color: {text_color}")
                logger.info(f"Adaptive text color: {text_color}")
            except Exception as e:
                print(f"⚠️ Adaptive colors failed, using default: {str(e)}")
                logger.warning(f"Adaptive colors failed, using default: {str(e)}")
                text_color = colors['text']
                contrast_ratio = 0.5  # Default contrast ratio
            
            # Draw each line
            shadow_offset = 2  # Reduced shadow offset for subtler effect
            border_width = 1   # Reduced border width for cleaner look
            
            for i, line in enumerate(lines):
                # Calculate position for this line
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                
                # Use smart X position if available, otherwise center
                try:
                    x = best_x - (line_width // 2)
                    # Ensure text stays within bounds
                    if x < 10:
                        x = 10
                    elif x + line_width > self.thumbnail_size[0] - 10:
                        x = self.thumbnail_size[0] - line_width - 10
                except:
                    x = (self.thumbnail_size[0] - line_width) // 2
                
                y = start_y + (i * line_height)
                
                # Add bold border for thumbnail visibility
                border_color = (255, 255, 255) if text_color[0] < 128 else (0, 0, 0)
                # Thick border for maximum visibility - 2 pixels
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx != 0 or dy != 0:
                            draw.text((x + dx, y + dy), line, font=font, fill=border_color)
                
                # Add main text using adaptive color (no shadow)
                draw.text((x, y), line, font=font, fill=text_color)
            
            # Save the final thumbnail
            filename = f"thumbnail_{hashlib.md5(text.encode()).hexdigest()}.png"
            thumbnail_path = os.path.join(self.temp_dir, filename)
            img.save(thumbnail_path, 'PNG', quality=95)
            
            logger.info(f"Thumbnail saved to: {thumbnail_path}")
            return thumbnail_path
            
        except Exception as e:
            logger.error(f"Error creating thumbnail with text: {str(e)}")
            return None
    
    def validate_text_quality(self, text: str) -> Tuple[bool, str]:
        """
        Validate text quality for the thumbnail.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text or not text.strip():
            return False, "Text cannot be empty"
        
        if len(text) > 100:
            return False, "Text too long for thumbnail display"
        
        # Check for common spelling issues (basic check)
        common_mistakes = {
            'teh': 'the',
            'recieve': 'receive',
            'seperate': 'separate',
            'definately': 'definitely'
        }
        
        text_lower = text.lower()
        for mistake, correction in common_mistakes.items():
            if mistake in text_lower:
                return False, f"Possible spelling error: '{mistake}' should be '{correction}'"
        
        return True, ""
