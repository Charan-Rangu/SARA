"""Sentinel-1 3-Channel SAR Preprocessing Engine for Small U-Net.

Converts 2-band calibrated Sentinel-1 SAR rasters (VV, VH) into the 3-channel
decibel input structure required by Small U-Net:
1. Band 1: VV_dB = 10 * log10(VV_linear)  [or VV_dB directly if already in dB]
2. Band 2: VH_dB = 10 * log10(VH_linear)  [or VH_dB directly if already in dB]
3. Band 3: VV_dB - VH_dB (Cross-polarization dB difference)

Features:
- Unit auto-detection preventing accidental double logarithmic conversion.
- Robust polarization band identification from metadata, tags, or explicit override.
- Strict error handling: fails if polarization identity cannot be established.
- Explicit invalid pixel tracking (nodata, NaN, Inf, <=0 linear power) mapped to nodata (255.0).
- Memory-efficient windowed block reading/writing (1024x1024) for multi-gigabyte rasters.
- Preservation of CRS, Affine Transform, spatial bounds, dimensions, and alignment.
- Normalization helper implementing verified Z-score parameters (means=[-11.0, -18.0, 7.0], stds=[3.0, 3.5, 2.5]).
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from pydantic import BaseModel, Field
from typing import Optional, Tuple, Dict, Any, List


class UNetPreprocessingConfig(BaseModel):
    """Configuration for 3-channel SAR preprocessing."""

    output_dir: str = os.path.join("data", "processed", "unet_inputs")
    block_size: int = 1024
    epsilon: float = 1e-7
    nodata_value: float = 255.0
    channel_means: List[float] = Field(default_factory=lambda: [-11.0, -18.0, 7.0])
    channel_stds: List[float] = Field(default_factory=lambda: [3.0, 3.5, 2.5])
    band_polarizations: Optional[List[str]] = None


class PreprocessingSummary(BaseModel):
    """Summary of 3-channel SAR preprocessing execution."""

    output_raster_path: str
    width: int
    height: int
    crs: str
    total_pixels: int
    valid_pixels: int
    nodata_pixels: int
    valid_percentage: float
    units_detected: str  # "linear" or "dB"
    band_mapping_used: str  # e.g. "VV=Band 2, VH=Band 1 -> Output Band 1=VV_dB, Band 2=VH_dB"
    band_means: List[float]
    band_stds: List[float]
    processing_time_seconds: float


class UNetSARPreprocessor:
    """Windowed 3-channel SAR preprocessor for Small U-Net input preparation."""

    def __init__(self, config: Optional[UNetPreprocessingConfig] = None):
        self.config = config or UNetPreprocessingConfig()

    @staticmethod
    def identify_polarizations(
        src: rasterio.DatasetReader,
        explicit_polarizations: Optional[Any] = None,
        input_path: Optional[str] = None
    ) -> Tuple[int, int]:
        """Identify 0-indexed band indices for (VV, VH) from explicit args, metadata, or sidecar provenance.

        Returns:
            Tuple[int, int]: (vv_idx, vh_idx)

        Raises:
            ValueError: If VV and VH cannot be reliably identified from metadata or config.
        """
        # 1. Explicit override (list, tuple, or dict)
        if explicit_polarizations is not None:
            if isinstance(explicit_polarizations, (list, tuple)):
                pols = [str(p).upper().strip() for p in explicit_polarizations]
                if "VV" in pols and "VH" in pols:
                    vv_idx = pols.index("VV")
                    vh_idx = pols.index("VH")
                    return vv_idx, vh_idx
            elif isinstance(explicit_polarizations, dict):
                pols = {str(k).upper().strip(): v for k, v in explicit_polarizations.items()}
                if "VV" in pols and "VH" in pols:
                    v_vv = pols["VV"]
                    v_vh = pols["VH"]
                    vv_idx = v_vv - 1 if v_vv >= 1 else v_vv
                    vh_idx = v_vh - 1 if v_vh >= 1 else v_vh
                    return vv_idx, vh_idx
            raise ValueError(
                f"Invalid explicit_polarizations format: {explicit_polarizations}. "
                "Expected list ['VV', 'VH'] or dict {'VV': 1, 'VH': 2}."
            )

        # 2. Check band descriptions
        if src.descriptions:
            desc_list = [str(d).upper() if d else "" for d in src.descriptions]
            vv_match = [i for i, d in enumerate(desc_list) if "VV" in d]
            vh_match = [i for i, d in enumerate(desc_list) if "VH" in d]
            if len(vv_match) == 1 and len(vh_match) == 1 and vv_match[0] != vh_match[0]:
                return vv_match[0], vh_match[0]

        # 3. Check band tags
        vv_idx, vh_idx = None, None
        for b in range(1, src.count + 1):
            tags = src.tags(b)
            tag_str = " ".join([f"{k}={v}" for k, v in tags.items()]).upper()
            if "VV" in tag_str and vv_idx is None:
                vv_idx = b - 1
            if "VH" in tag_str and vh_idx is None:
                vh_idx = b - 1

        if vv_idx is not None and vh_idx is not None and vv_idx != vh_idx:
            return vv_idx, vh_idx

        # 4. Check global tags
        gtags = src.tags()
        gtag_str = " ".join([f"{k}={v}" for k, v in gtags.items()]).upper()
        if "BAND_1=VV" in gtag_str or "POLARIZATION_1=VV" in gtag_str or "POLARISATIONS=VV,VH" in gtag_str:
            return 0, 1
        if "BAND_1=VH" in gtag_str or "POLARIZATION_1=VH" in gtag_str or "POLARISATIONS=VH,VV" in gtag_str:
            return 1, 0

        # 5. Check sidecar metadata file or SNAP graph file if input_path provided
        if input_path:
            base = os.path.splitext(input_path)[0]
            sidecar_json = f"{base}_metadata.json"
            if os.path.exists(sidecar_json):
                import json
                try:
                    with open(sidecar_json, "r") as f:
                        sdata = json.load(f)
                    p_list = sdata.get("polarizations") or sdata.get("band_order")
                    if p_list:
                        pols = [str(p).upper() for p in p_list]
                        if "VV" in pols and "VH" in pols:
                            return pols.index("VV"), pols.index("VH")
                except Exception:
                    pass

            # Check if SNAP post_process_graph.xml exists in parent/same directory
            graph_xml = os.path.join(os.path.dirname(input_path), "post_process_graph.xml")
            if os.path.exists(graph_xml):
                try:
                    with open(graph_xml, "r") as f:
                        xml_content = f.read()
                    if "<selectedPolarisations>VV,VH</selectedPolarisations>" in xml_content:
                        # SNAP GPT output for S1 GRDH writes VH as Band 1 and VV as Band 2
                        return 1, 0
                except Exception:
                    pass

        # 6. Strict failure if polarization cannot be identified
        raise ValueError(
            f"Unable to reliably identify VV and VH polarization bands from raster metadata or provenance for '{input_path}'. "
            "Please specify explicit 'band_polarizations' (e.g. ['VH', 'VV']) or ensure input band descriptions contain VV/VH."
        )

    @staticmethod
    def detect_units(sample_array: np.ndarray, nodata_val: Optional[float] = None) -> str:
        """Detect whether sample SAR array is in linear power scale (sigma0) or decibels (dB).

        Linear power scale: positive values, typically 0.0001 to 10.0, mean < 1.0.
        Decibel scale: includes negative decibel values (e.g. -45 dB to +10 dB).
        """
        valid_mask = np.isfinite(sample_array)
        if nodata_val is not None:
            valid_mask &= (sample_array != nodata_val)

        valid_vals = sample_array[valid_mask]
        if len(valid_vals) == 0:
            return "linear"

        # Filter plausible non-zero values
        non_zero = valid_vals[valid_vals != 0.0]
        if len(non_zero) == 0:
            return "linear"

        # If significant negative values (< -2.0 dB) exist, it is decibels
        if np.any(non_zero < -2.0) or np.mean(non_zero) < 0.0:
            return "dB"

        pos_vals = non_zero[non_zero > 0.0]
        if len(pos_vals) > 0 and np.mean(pos_vals) < 2.0:
            return "linear"

        return "dB"

    def process_raster_file(
        self,
        input_raster_path: str,
        output_raster_path: Optional[str] = None,
        force_units: Optional[str] = None,
        band_polarizations: Optional[Any] = None
    ) -> PreprocessingSummary:
        """Process 2-band or 3-band SAR raster into memory-efficient 3-band Small U-Net ARD format.

        Args:
            input_raster_path: Path to input Sentinel-1 GeoTIFF.
            output_raster_path: Path for target 3-band GeoTIFF.
            force_units: Optional explicit override ("linear" or "dB").
            band_polarizations: Optional explicit polarization band order (e.g. ['VH', 'VV'] or {'VV': 2, 'VH': 1}).

        Returns:
            PreprocessingSummary object.
        """
        import time
        start_time = time.time()

        if not os.path.exists(input_raster_path):
            raise FileNotFoundError(f"Input SAR raster not found: {input_raster_path}")

        if output_raster_path is None:
            os.makedirs(self.config.output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(input_raster_path))[0]
            output_raster_path = os.path.join(self.config.output_dir, f"{base_name}_unet_3band.tif")
        else:
            out_dir = os.path.dirname(os.path.abspath(output_raster_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        with rasterio.open(input_raster_path) as src:
            if src.count < 2:
                raise ValueError(f"Input SAR raster must contain at least 2 bands (VV, VH), got {src.count} bands.")

            width = src.width
            height = src.height
            total_pixels = width * height
            profile = src.profile.copy()
            nodata_in = src.nodata

            # Explicitly identify polarization band mapping
            effective_pols = band_polarizations if band_polarizations is not None else self.config.band_polarizations
            vv_idx, vh_idx = self.identify_polarizations(src, explicit_polarizations=effective_pols, input_path=input_raster_path)
            band_mapping_str = f"VV=Band {vv_idx+1}, VH=Band {vh_idx+1} -> Output Band 1=VV_dB, Band 2=VH_dB, Band 3=VV_dB - VH_dB"

            # Sample center block to detect input units
            win_sample = Window(max(0, width // 2 - 256), max(0, height // 2 - 256), min(512, width), min(512, height))
            sample_data = src.read(window=win_sample)
            
            detected_units = force_units or self.detect_units(sample_data[vv_idx], nodata_val=nodata_in)

            # Configure output GeoTIFF profile (enabling BIGTIFF for large multi-gigabyte rasters)
            profile.update(
                count=3,
                dtype=rasterio.float32,
                nodata=self.config.nodata_value,
                driver="GTiff",
                BIGTIFF="IF_SAFER",
                compress="lzw"
            )

            valid_pixel_count = 0
            nodata_pixel_count = 0

            # Accumulators for online statistics
            sum_b1, sum_b2, sum_b3 = 0.0, 0.0, 0.0
            sq_b1, sq_b2, sq_b3 = 0.0, 0.0, 0.0

            block_size = self.config.block_size

            with rasterio.open(output_raster_path, "w", **profile) as dst:
                dst.set_band_description(1, "VV_dB")
                dst.set_band_description(2, "VH_dB")
                dst.set_band_description(3, "VV_dB - VH_dB")

                dst.update_tags(1, POLARIZATION="VV", UNITS="dB")
                dst.update_tags(2, POLARIZATION="VH", UNITS="dB")
                dst.update_tags(3, POLARIZATION="VV-VH", UNITS="dB")

                for y in range(0, height, block_size):
                    for x in range(0, width, block_size):
                        win_w = min(block_size, width - x)
                        win_h = min(block_size, height - y)
                        win = Window(x, y, win_w, win_h)

                        chip = src.read(window=win)  # (count, win_h, win_w)
                        vv_raw = chip[vv_idx].astype(np.float32)
                        vh_raw = chip[vh_idx].astype(np.float32)

                        # Explicit invalid pixel tracking
                        invalid_mask = np.isnan(vv_raw) | np.isnan(vh_raw) | np.isinf(vv_raw) | np.isinf(vh_raw)
                        if nodata_in is not None:
                            invalid_mask |= (vv_raw == nodata_in) | (vh_raw == nodata_in)

                        if detected_units == "linear":
                            invalid_mask |= (vv_raw <= 0.0) | (vh_raw <= 0.0)

                        valid_mask = ~invalid_mask

                        # Allocate output block
                        out_block = np.full((3, win_h, win_w), fill_value=self.config.nodata_value, dtype=np.float32)

                        if np.any(valid_mask):
                            vv_v = vv_raw[valid_mask]
                            vh_v = vh_raw[valid_mask]

                            if detected_units == "linear":
                                vv_v_db = 10.0 * np.log10(np.clip(vv_v, self.config.epsilon, None))
                                vh_v_db = 10.0 * np.log10(np.clip(vh_v, self.config.epsilon, None))
                            else:
                                vv_v_db = vv_v
                                vh_v_db = vh_v

                            diff_v_db = vv_v_db - vh_v_db

                            out_block[0][valid_mask] = vv_v_db
                            out_block[1][valid_mask] = vh_v_db
                            out_block[2][valid_mask] = diff_v_db

                            # Stats accumulation
                            n_v = np.sum(valid_mask)
                            valid_pixel_count += int(n_v)
                            nodata_pixel_count += int(np.sum(invalid_mask))

                            sum_b1 += float(np.sum(vv_v_db))
                            sum_b2 += float(np.sum(vh_v_db))
                            sum_b3 += float(np.sum(diff_v_db))

                            sq_b1 += float(np.sum(vv_v_db ** 2))
                            sq_b2 += float(np.sum(vh_v_db ** 2))
                            sq_b3 += float(np.sum(diff_v_db ** 2))
                        else:
                            nodata_pixel_count += (win_w * win_h)

                        dst.write(out_block, window=win)

        exec_time = time.time() - start_time

        if valid_pixel_count > 0:
            m1 = sum_b1 / valid_pixel_count
            m2 = sum_b2 / valid_pixel_count
            m3 = sum_b3 / valid_pixel_count

            v1 = max(0.0, (sq_b1 / valid_pixel_count) - (m1 ** 2))
            v2 = max(0.0, (sq_b2 / valid_pixel_count) - (m2 ** 2))
            v3 = max(0.0, (sq_b3 / valid_pixel_count) - (m3 ** 2))

            std1, std2, std3 = np.sqrt(v1), np.sqrt(v2), np.sqrt(v3)
            b_means = [round(m1, 4), round(m2, 4), round(m3, 4)]
            b_stds = [round(std1, 4), round(std2, 4), round(std3, 4)]
        else:
            b_means = [0.0, 0.0, 0.0]
            b_stds = [0.0, 0.0, 0.0]

        valid_pct = (valid_pixel_count / total_pixels * 100.0) if total_pixels > 0 else 0.0

        return PreprocessingSummary(
            output_raster_path=output_raster_path,
            width=width,
            height=height,
            crs=str(profile.get("crs", "")),
            total_pixels=total_pixels,
            valid_pixels=valid_pixel_count,
            nodata_pixels=nodata_pixel_count,
            valid_percentage=round(valid_pct, 4),
            units_detected=detected_units,
            band_mapping_used=band_mapping_str,
            band_means=b_means,
            band_stds=b_stds,
            processing_time_seconds=round(exec_time, 4)
        )

    def apply_unet_normalization(self, input_array: np.ndarray) -> np.ndarray:
        """Apply Small U-Net channel-wise Z-score normalization to a 3-channel image array.

        Args:
            input_array: Numpy array of shape (3, H, W) in decibels.

        Returns:
            Normalized array of shape (3, H, W).
        """
        if input_array.shape[0] != 3:
            raise ValueError(f"Normalization requires 3-channel input array, got shape {input_array.shape}")

        means = np.array(self.config.channel_means, dtype=np.float32).reshape(3, 1, 1)
        stds = np.array(self.config.channel_stds, dtype=np.float32).reshape(3, 1, 1)

        return (input_array - means) / (stds + 1e-7)
