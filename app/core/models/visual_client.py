"""Client for visual analysis using OpenAI GPT-4 Vision."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from PIL import Image
import io

# Disable DecompressionBombError for large satellite images
Image.MAX_IMAGE_PIXELS = None


VISUAL_ANALYSIS_SYSTEM_PROMPT = """
You are an expert disaster damage analyst specializing in aerial and satellite imagery interpretation.

## CRITICAL TEMPORAL CONTEXT:
- Hurricane Harvey PEAK FLOODING: August 27-28, 2017
- This satellite imagery was captured: August 31, 2017 (3-4 days AFTER peak)
- By August 31, most FLOODWATERS had RECEDED from streets and buildings
- You are looking at POST-FLOOD imagery, not during-flood imagery

## What to Look For (POST-FLOOD INDICATORS):
1. **Evidence of Past Flooding** (not current water):
   - Water line marks/stains on buildings
   - Mud, sediment, or debris deposits on streets
   - Discolored vegetation or dead grass
   - Debris piles near homes or in yards
   
2. **Structural Damage** (PERSISTENT, visible days later):
   - Damaged roofs (missing shingles, tarps, holes)
   - Collapsed or leaning structures
   - Debris scattered around properties
   - Damaged fences or outbuildings

3. **Infrastructure Issues**:
   - Road damage, potholes, debris on roads
   - Utility damage (downed lines visible)

## IMPORTANT:
- "No standing water visible" does NOT mean flooding didn't occur - water RECEDED
- Focus on DAMAGE EVIDENCE and DEBRIS rather than looking for current water
- If you see clean, undamaged areas → that's useful info (no damage in this area)

Respond with valid JSON only.
"""


def build_visual_analysis_prompt(
    zip_code: str,
    time_window: Dict[str, str],
    tile_ids: List[str],
    text_summary: str = None,
) -> str:
    """Build the user prompt for visual analysis, optionally guided by text analysis."""

    # Text guidance section (if text analysis results are provided)
    if text_summary:
        text_guidance = f"""
## TEXT ANALYSIS SUMMARY (from tweets/311 calls):
{text_summary}

Your task: Look for VISUAL CONFIRMATION of the above text reports.
- If text says "flooding reported" → look for water lines, debris, mud
- If text says "house damaged" → look for roof damage, structural issues
- If you see evidence CONFIRMING text → note "CONFIRMED by imagery"
- If you see NO evidence → note "NOT VISIBLE in imagery" (doesn't mean it didn't happen)
"""
    else:
        text_guidance = ""

    return f"""
Analyze the following POST-FLOOD satellite imagery (captured Aug 31, 2017).

Location: ZIP {zip_code}
Time Period: {time_window['start']} to {time_window['end']}
Image Tiles: {tile_ids}
{text_guidance}
For each image, look for POST-FLOOD evidence:
- Debris, sediment, water stains (evidence of past flooding)
- Structural damage (roofs, walls, collapsed buildings)
- Road/infrastructure damage

Respond with JSON matching this schema:
{{
    "image_observations": [
        {{
            "tile_id": str,
            "flood_evidence_visible": bool,  // Debris, water lines, sediment
            "flood_evidence_description": str | null,
            "structural_damage_visible": bool,
            "damage_description": str | null,
            "confirms_text_reports": bool,  // Does this confirm text analysis?
            "confidence": float  // 0.0 to 1.0
        }}
    ],
    "overall_assessment": {{
        "flood_evidence_pct": float,  // 0-100: How much area shows flood evidence?
        "structural_damage_pct": float,  // 0-100: Visible damage
        "text_confirmation_level": str,  // "strong", "partial", "none", "contradicts"
        "key_observations": list[str],
        "confidence": float  // 0.0 to 1.0
    }},
    "evidence_refs": {{
        "imagery_tile_ids": list[str]  // IDs of tiles that show damage
    }}
}}
"""


class VisualAnalysisClient:
    """Client for analyzing imagery using OpenAI GPT-4 Vision or Google Gemini."""

    def __init__(
        self,
        model_name: str = None,
        api_key: Optional[str] = None,
        provider: str = None,
        max_images: int = 6,
        max_image_size: int = 1024,
    ):
        """
        Initialize the visual analysis client.

        Args:
            model_name: Model to use (default: gpt-4o for OpenAI, gemini-2.0-flash-exp for Gemini)
            api_key: API key (OpenAI or Gemini)
            provider: "openai" or "gemini" (auto-detected from env if not provided)
            max_images: Maximum number of images to send per request
            max_image_size: Maximum dimension for image resizing
        """
        self.provider = provider or os.getenv("VISUAL_MODEL_PROVIDER", "openai")
        self.max_images = max_images
        self.max_image_size = max_image_size

        if self.provider == "gemini":
            self._init_gemini(model_name, api_key or os.getenv("GEMINI_API_KEY"))
        else:
            self._init_openai(model_name, api_key or os.getenv("OPENAI_API_KEY"))

    def _init_openai(self, model_name: str, api_key: str):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        if not api_key:
            logger.warning("OPENAI_API_KEY not set. VisualAnalysisClient will fail.")

        self.model_name = model_name or os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
        self.client = OpenAI(api_key=api_key)
        logger.info(
            f"VisualAnalysisClient initialized with OpenAI model: {self.model_name}"
        )

    def _init_gemini(self, model_name: str, api_key: str):
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. Run: pip install google-generativeai"
            )

        if not api_key:
            logger.warning("GEMINI_API_KEY not set. VisualAnalysisClient will fail.")

        genai.configure(api_key=api_key)
        self.model_name = model_name or os.getenv(
            "GEMINI_VISION_MODEL", "gemini-2.0-flash-exp"
        )
        self.genai = genai
        self.gemini_model = genai.GenerativeModel(self.model_name)
        logger.info(
            f"VisualAnalysisClient initialized with Gemini model: {self.model_name}"
        )

    def analyze(
        self,
        zip_code: str,
        time_window: Dict[str, str],
        imagery_tiles: List[Dict],
        project_root: Path = None,
        text_summary: str = None,
    ) -> Dict[str, Any]:
        """
        Analyze imagery tiles for disaster damage.

        Args:
            zip_code: Target ZIP code
            time_window: Dict with 'start' and 'end' dates
            imagery_tiles: List of tile metadata dicts with 'tile_id' and 'uri'
            project_root: Project root for resolving relative paths
            text_summary: Optional summary from text analysis to guide visual search

        Returns:
            Analysis results with observations and damage estimates
        """
        if not imagery_tiles:
            logger.warning("No imagery tiles provided for analysis")
            return self._empty_result()

        project_root = project_root or Path.cwd()

        # Limit number of images
        tiles_to_analyze = imagery_tiles[: self.max_images]
        tile_ids = [t.get("tile_id", "unknown") for t in tiles_to_analyze]

        logger.info(
            f"Analyzing {len(tiles_to_analyze)} imagery tiles for ZIP {zip_code}"
        )

        # Load images (format depends on provider)
        image_data = []
        valid_tile_ids = []

        for tile in tiles_to_analyze:
            uri = tile.get("uri")
            tile_id = tile.get("tile_id", "unknown")

            if not uri:
                logger.warning(f"Tile {tile_id} has no URI, skipping")
                continue

            # Resolve path
            uri_path = Path(uri)
            if uri_path.is_absolute():
                image_path = uri_path
            else:
                image_path = project_root / uri

            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            try:
                if self.provider == "gemini":
                    # Gemini uses PIL Image objects
                    img = Image.open(image_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    # Resize if needed
                    if max(img.size) > self.max_image_size:
                        ratio = self.max_image_size / max(img.size)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    image_data.append(img)
                else:
                    # OpenAI uses base64
                    base64_image = self._encode_image(image_path)
                    image_data.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high",
                            },
                        }
                    )
                valid_tile_ids.append(tile_id)
            except Exception as e:
                logger.error(f"Failed to load image {tile_id}: {e}")
                continue

        if not image_data:
            logger.warning("No valid images could be loaded")
            return self._empty_result()

        # Build the prompt (with text guidance if available)
        prompt = build_visual_analysis_prompt(
            zip_code, time_window, valid_tile_ids, text_summary
        )

        try:
            if self.provider == "gemini":
                # Gemini API
                full_prompt = f"{VISUAL_ANALYSIS_SYSTEM_PROMPT}\n\n{prompt}"
                # Gemini can take multiple images
                content = [full_prompt] + image_data
                response = self.gemini_model.generate_content(
                    content,
                    generation_config={
                        "response_mime_type": "application/json",
                        "max_output_tokens": 2000,
                    },
                )
                result = json.loads(response.text)
            else:
                # OpenAI API
                user_content = [{"type": "text", "text": prompt}] + image_data
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": VISUAL_ANALYSIS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=2000,
                )
                result = json.loads(response.choices[0].message.content)

            logger.info(
                f"Visual analysis complete: "
                f"flood={result.get('overall_assessment', {}).get('flood_evidence_pct', result.get('overall_assessment', {}).get('flood_severity_pct', 0)):.1f}%, "
                f"damage={result.get('overall_assessment', {}).get('structural_damage_pct', 0):.1f}%"
            )
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse visual analysis response: {e}")
            return self._empty_result()
        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            return self._empty_result()

    def _encode_image(self, image_path: Path) -> str:
        """Load and encode an image as base64, resizing if needed."""
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize if too large
            if max(img.size) > self.max_image_size:
                ratio = self.max_image_size / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Encode to base64
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _empty_result(self) -> Dict[str, Any]:
        """Return an empty result structure."""
        return {
            "image_observations": [],
            "overall_assessment": {
                "flood_severity_pct": 0.0,
                "structural_damage_pct": 0.0,
                "key_observations": ["No imagery could be analyzed"],
                "confidence": 0.0,
            },
            "evidence_refs": {"imagery_tile_ids": []},
        }

    def analyze_single(
        self,
        image_path: Path,
        tile_id: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Analyze a single image for quick testing.

        Args:
            image_path: Path to the image file
            tile_id: Identifier for the tile

        Returns:
            Analysis results for the single image
        """
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return self._empty_result()

        try:
            base64_image = self._encode_image(image_path)
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return self._empty_result()

        prompt = f"""
Analyze this disaster imagery tile (ID: {tile_id}).

Describe what you see in terms of:
1. Flooding (standing water, submerged areas)
2. Structural damage (roof damage, collapsed buildings)
3. Infrastructure issues (road conditions, debris)

Respond with JSON:
{{
    "tile_id": "{tile_id}",
    "flooding_visible": bool,
    "flooding_description": str | null,
    "structural_damage_visible": bool,
    "damage_description": str | null,
    "infrastructure_issues": str | null,
    "severity_pct": float,  // 0-100
    "confidence": float  // 0.0-1.0
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": VISUAL_ANALYSIS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Single image analysis failed: {e}")
            return {
                "tile_id": tile_id,
                "error": str(e),
                "flooding_visible": False,
                "structural_damage_visible": False,
                "severity_pct": 0.0,
                "confidence": 0.0,
            }
