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
Your task is to analyze post-disaster imagery from Hurricane Harvey (August-September 2017) in Harris County, TX.

For each image, you should identify and describe:
1. **Flooding**: Standing water, submerged roads, inundation extent
2. **Structural Damage**: Damaged roofs, collapsed structures, debris
3. **Infrastructure**: Road conditions, utility damage, blocked access
4. **Severity Assessment**: Rate the visible damage on a scale of 0-100%

CRITICAL RULES:
- ONLY describe what you can DIRECTLY SEE in the images
- Do NOT infer or assume damage that is not visible
- If an image is unclear or shows no damage, say so explicitly
- Reference specific image IDs when describing observations

Respond with valid JSON only.
"""


def build_visual_analysis_prompt(
    zip_code: str,
    time_window: Dict[str, str],
    tile_ids: List[str],
) -> str:
    """Build the user prompt for visual analysis."""
    return f"""
Analyze the following disaster imagery for damage assessment.

Location: ZIP {zip_code}
Time Period: {time_window['start']} to {time_window['end']}
Image Tiles: {tile_ids}

For each image, describe what you observe. Then provide an overall assessment.

Respond with JSON matching this schema:
{{
    "image_observations": [
        {{
            "tile_id": str,
            "flooding_visible": bool,
            "flooding_description": str | null,
            "structural_damage_visible": bool,
            "damage_description": str | null,
            "infrastructure_issues": str | null,
            "confidence": float  // 0.0 to 1.0
        }}
    ],
    "overall_assessment": {{
        "flood_severity_pct": float,  // 0-100
        "structural_damage_pct": float,  // 0-100
        "key_observations": list[str],
        "confidence": float  // 0.0 to 1.0
    }},
    "evidence_refs": {{
        "imagery_tile_ids": list[str]  // IDs of tiles that show damage
    }}
}}
"""


class VisualAnalysisClient:
    """Client for analyzing imagery using OpenAI GPT-4 Vision."""

    def __init__(
        self,
        model_name: str = None,
        api_key: Optional[str] = None,
        max_images: int = 6,
        max_image_size: int = 1024,
    ):
        """
        Initialize the visual analysis client.

        Args:
            model_name: OpenAI model to use (default: gpt-4o)
            api_key: OpenAI API key
            max_images: Maximum number of images to send per request
            max_image_size: Maximum dimension for image resizing
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set. VisualAnalysisClient will fail.")

        self.model_name = model_name or os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
        self.client = OpenAI(api_key=self.api_key)
        self.max_images = max_images
        self.max_image_size = max_image_size

        logger.info(f"VisualAnalysisClient initialized with model: {self.model_name}")

    def analyze(
        self,
        zip_code: str,
        time_window: Dict[str, str],
        imagery_tiles: List[Dict],
        project_root: Path = None,
    ) -> Dict[str, Any]:
        """
        Analyze imagery tiles for disaster damage.

        Args:
            zip_code: Target ZIP code
            time_window: Dict with 'start' and 'end' dates
            imagery_tiles: List of tile metadata dicts with 'tile_id' and 'uri'
            project_root: Project root for resolving relative paths

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

        logger.info(f"Analyzing {len(tiles_to_analyze)} imagery tiles for ZIP {zip_code}")

        # Load and encode images
        image_contents = []
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
                base64_image = self._encode_image(image_path)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high",
                    },
                })
                valid_tile_ids.append(tile_id)
            except Exception as e:
                logger.error(f"Failed to encode image {tile_id}: {e}")
                continue

        if not image_contents:
            logger.warning("No valid images could be loaded")
            return self._empty_result()

        # Build the prompt
        prompt = build_visual_analysis_prompt(zip_code, time_window, valid_tile_ids)

        # Create message with images
        user_content = [{"type": "text", "text": prompt}] + image_contents

        try:
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
                f"flood={result.get('overall_assessment', {}).get('flood_severity_pct', 0):.1f}%, "
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

