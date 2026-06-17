"""HRRR 环境场读取与时空插值模块。

从 HRRR (High-Resolution Rapid Refresh) 分析场 NetCDF 文件中提取
u, v, w, T, ρ, p 等三维大气变量，通过双线性水平插值 + 线性垂直插值
将 HRRR 数据映射到 GridRad 雷达网格上。

支持时间插值：HRRR 每小时一次，GridRad 每 5 分钟一次，
非整点时刻通过前后整点线性时间插值得到。

参考文献：
  - Benjamin et al. (2016) "A North American Hourly Assimilation and
    Model Forecast Cycle: The Rapid Refresh"
  - GridRad-Severe 数据集使用说明
"""
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any
from functools import lru_cache

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False
    Dataset = None

try:
    from scipy.interpolate import RegularGridInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    RegularGridInterpolator = None


# HRRR 变量名映射（支持多种命名约定）
_HRRR_VAR_MAP = {
    "u": ["u", "U", "ugrd10m", "UGRD"],
    "v": ["v", "V", "vgrd10m", "VGRD"],
    "w": ["w", "W", "dz_dt", "DZDT"],
    "t": ["t", "T", "tmp", "TMP"],
    "q": ["q", "Q", "spfh", "SPFH", "Q"],
    "p": ["p", "P", "pres", "PRES", "pressure"],
    "rho": ["rho", "RHO", "dens", "DENS"],
}

# 标准等压面层 (hPa)，HRRR 典型输出
_STANDARD_PRESSURE_LEVELS = np.array([
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
    750, 725, 700, 650, 600, 550, 500, 450, 400, 350,
    300, 250, 200, 150, 100, 70, 50, 30, 20, 10,
    7, 5, 3, 2, 1, 0.5,
], dtype=np.float64)


def _resolve_var_name(ds_variables: set, canonical: str) -> str | None:
    """在数据集的变量名中找到 canonical 变量对应的实际名称。"""
    candidates = _HRRR_VAR_MAP.get(canonical, [canonical])
    for name in candidates:
        if name in ds_variables:
            return name
    return None


def _compute_rho(pressure_hpa: np.ndarray, temp_k: np.ndarray,
                 specific_humidity: np.ndarray) -> np.ndarray:
    """从 p, T, q 计算湿空气密度 ρ (kg/m³)。

    ρ = p / (R_d * T_v)
    其中 R_d = 287.058 J/(kg·K)，T_v = T * (1 + 0.608 * q)

    Args:
        pressure_hpa: 气压 (hPa)
        temp_k: 温度 (K)
        specific_humidity: 比湿 (kg/kg)

    Returns:
        密度 (kg/m³)
    """
    R_d = 287.058
    pressure_pa = pressure_hpa * 100.0  # hPa → Pa
    T_v = temp_k * (1.0 + 0.608 * specific_humidity)
    return pressure_pa / (R_d * T_v + 1e-8)


def _height_from_pressure(pressure_hpa: np.ndarray) -> np.ndarray:
    """从气压近似计算海拔高度 (m)，使用标准大气压高公式。

    仅在没有直接高度信息时使用。
    """
    P0 = 1013.25  # 海平面标准气压 hPa
    scale_height = 7400.0  # 大气标高 m
    return scale_height * np.log(P0 / np.maximum(pressure_hpa, 1e-3))


class HRRRReader:
    """HRRR 分析场时空插值读取器。

    使用方式:
        reader = HRRRReader("/path/to/hrrr/")
        fields = reader.get_fields(
            timestamp, grid_lon, grid_lat, grid_alt,
            variables=["u", "v", "w", "t", "q", "p"]
        )
        # fields["u"]: (H, W, L) 数组
    """

    def __init__(self, hrrr_dir: str):
        if not HAS_NETCDF4:
            raise ImportError(
                "HRRR Reader 需要 netCDF4: pip install netCDF4"
            )
        if not HAS_SCIPY:
            raise ImportError(
                "HRRR Reader 需要 scipy: pip install scipy"
            )

        self.hrrr_dir = Path(hrrr_dir)
        self._index: dict[int, Path] = {}
        self._coord_cache: dict[str, np.ndarray] = {}
        self._build_index()

    def _build_index(self):
        """扫描 HRRR 目录，建立 {unix_timestamp: filepath} 索引。"""
        for f in sorted(self.hrrr_dir.glob("*.nc")):
            # 尝试从文件名解析时间戳
            import re
            ts = None
            # 格式: hrrr.t00z.wrfprsf00.nc 或类似
            m = re.search(r"(\d{8}T\d{6}Z?)", f.name)
            if m:
                try:
                    dt = datetime.strptime(m.group(1).rstrip("Z"), "%Y%m%dT%H%M")
                    ts = int(dt.timestamp())
                except ValueError:
                    pass
            # 格式: YYYYMMDDHH.nc
            m = re.search(r"(\d{10})", f.name)
            if m and ts is None:
                try:
                    dt = datetime.strptime(m.group(1), "%Y%m%d%H")
                    ts = int(dt.timestamp())
                except ValueError:
                    pass
            # 格式: hrrr_YYYYMMDD_HH.nc
            m = re.search(r"(\d{8})_(\d{2})", f.name)
            if m and ts is None:
                try:
                    dt = datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H")
                    ts = int(dt.timestamp())
                except ValueError:
                    pass

            if ts is not None:
                self._index[ts] = f

        if not self._index:
            # 退化：按修改时间索引
            for f in sorted(self.hrrr_dir.glob("*.nc")):
                self._index[int(f.stat().st_mtime)] = f

    @property
    def is_available(self) -> bool:
        """HRRR 数据是否可用。"""
        return len(self._index) > 0

    def _load_coords(self, nc_path: Path) -> dict[str, np.ndarray]:
        """从 NetCDF 文件中加载坐标变量（缓存）。"""
        cache_key = str(nc_path)
        if cache_key in self._coord_cache:
            return self._coord_cache[cache_key]

        with Dataset(str(nc_path), "r") as ds:
            coords = {}
            # 经纬度
            for name in ["longitude", "Longitude", "lon", "LON",
                         "XLONG", "longitude_0"]:
                if name in ds.variables:
                    coords["lon"] = ds.variables[name][:].astype(np.float64)
                    break
            for name in ["latitude", "Latitude", "lat", "LAT",
                         "XLAT", "latitude_0"]:
                if name in ds.variables:
                    coords["lat"] = ds.variables[name][:].astype(np.float64)
                    break

            # 垂直坐标
            for name in ["pressure", "isobaric", "lev", "level",
                         "pres", "isobaric1"]:
                if name in ds.variables:
                    coords["pressure"] = ds.variables[name][:].astype(np.float64)
                    break
            if "pressure" not in coords:
                coords["pressure"] = _STANDARD_PRESSURE_LEVELS.copy()

            # 若为 2D lat/lon（WRF 风格），取均值转为 1D
            if coords.get("lat") is not None and coords["lat"].ndim == 2:
                coords["lat"] = coords["lat"].mean(axis=1)
            if coords.get("lon") is not None and coords["lon"].ndim == 2:
                coords["lon"] = coords["lon"].mean(axis=0)

        self._coord_cache[cache_key] = coords
        return coords

    def _read_fields(self, nc_path: Path,
                     variables: list[str]) -> dict[str, np.ndarray]:
        """从单个 HRRR 文件中读取指定变量。"""
        with Dataset(str(nc_path), "r") as ds:
            all_vars = set(ds.variables.keys())
            fields = {}
            for var in variables:
                actual_name = _resolve_var_name(all_vars, var)
                if actual_name is None:
                    continue
                data = ds.variables[actual_name][:].astype(np.float32)
                # 处理维度：挤压单例维度
                while data.ndim > 3 and data.shape[0] == 1:
                    data = data.squeeze(0)
                fields[var] = data
            return fields

    def get_fields(self, timestamp: datetime,
                   grid_lon: np.ndarray,
                   grid_lat: np.ndarray,
                   grid_alt: np.ndarray,
                   variables: list[str],
                   ) -> dict[str, np.ndarray]:
        """获取指定时间戳的 HRRR 变量，插值到 GridRad 网格。

        Args:
            timestamp: 请求时间 (UTC)
            grid_lon: GridRad 目标经度数组 (W,)，1D
            grid_lat: GridRad 目标纬度数组 (H,)，1D
            grid_alt: GridRad 目标高度数组 (L,)，1D（米）
            variables: 请求的变量名列表，如 ["u","v","w","t","q","p"]

        Returns:
            {var_name: (H, W, L) ndarray}，缺失的变量缺失键
        """
        if not self._index:
            return {}

        # 找到前后最近的 HRRR 文件
        target_ts = int(timestamp.timestamp())
        ts_list = sorted(self._index.keys())
        idx = np.searchsorted(ts_list, target_ts)

        # 时间插值权重
        if idx == 0:
            # 目标时间在所有文件之前 → 使用第一个文件
            files_ts = [(ts_list[0], 1.0)]
        elif idx >= len(ts_list):
            # 目标时间在所有文件之后 → 使用最后一个文件
            files_ts = [(ts_list[-1], 1.0)]
        else:
            t_before = ts_list[idx - 1]
            t_after = ts_list[idx]
            alpha = (target_ts - t_before) / max(t_after - t_before, 1)
            files_ts = [(t_before, 1.0 - alpha), (t_after, alpha)]

        # 对每个时间点读取并插值
        accumulated: dict[str, np.ndarray] = {}
        for ts, weight in files_ts:
            nc_path = self._index[ts]
            coords = self._load_coords(nc_path)
            fields = self._read_fields(nc_path, variables)

            # 如果缺少 rho，从 p, t, q 计算
            if "rho" in variables and "rho" not in fields:
                if all(k in fields for k in ["p", "t", "q"]):
                    fields["rho"] = _compute_rho(
                        fields["p"], fields["t"], fields["q"]
                    )

            for var in variables:
                if var not in fields:
                    continue
                interpolated = self._interpolate_3d(
                    fields[var], coords, grid_lon, grid_lat, grid_alt,
                )
                if var not in accumulated:
                    accumulated[var] = weight * interpolated
                else:
                    accumulated[var] += weight * interpolated

        return accumulated

    def _interpolate_3d(self, field: np.ndarray,
                        coords: dict[str, np.ndarray],
                        grid_lon: np.ndarray,
                        grid_lat: np.ndarray,
                        grid_alt: np.ndarray,
                        ) -> np.ndarray:
        """将 HRRR 场插值到 GridRad 网格。

        1. 水平双线性插值 (HRRR lon/lat → GridRad H×W)
        2. 垂直线性插值 (HRRR 气压层 → GridRad 高度层)

        Args:
            field: HRRR 场 (L_hrrr, H_hrrr, W_hrrr)
            coords: HRRR 坐标 {"lon": (W_hrrr,), "lat": (H_hrrr,), "pressure": (L_hrrr,)}
            grid_lon: 目标经度 (W,)
            grid_lat: 目标纬度 (H,)
            grid_alt: 目标高度 (L,) 米

        Returns:
            插值后场 (H, W, L)
        """
        L_hrrr, H_hrrr, W_hrrr = field.shape[-3:]
        H_out, W_out, L_out = len(grid_lat), len(grid_lon), len(grid_alt)

        # 垂直坐标：HRRR 气压层 → 高度 (m)
        hrrr_pressure = coords.get("pressure",
                                    _STANDARD_PRESSURE_LEVELS[:L_hrrr])
        hrrr_alt = _height_from_pressure(hrrr_pressure)
        if len(hrrr_alt) != L_hrrr:
            hrrr_alt = hrrr_alt[:L_hrrr]

        hrrr_lat = coords.get("lat", np.linspace(20, 55, H_hrrr))
        hrrr_lon = coords.get("lon", np.linspace(-130, -60, W_hrrr))

        # --- 第一步：垂直插值 (对各 HRRR 格点，从气压层插值到目标高度) ---
        # 构建垂直插值器：对于每个 (i, j)，field[:, i, j] 在 hrrr_alt 上插值
        # 为了提高效率，直接用 np.interp 对每个水平格点
        field_vert = np.zeros((L_out, H_hrrr, W_hrrr), dtype=np.float32)
        for i in range(H_hrrr):
            for j in range(W_hrrr):
                profile = field[:, i, j]
                field_vert[:, i, j] = np.interp(
                    grid_alt, hrrr_alt, profile,
                    left=profile[0], right=profile[-1],
                )

        # --- 第二步：水平双线性插值 (对每个高度层) ---
        result = np.zeros((H_out, W_out, L_out), dtype=np.float32)
        for k in range(L_out):
            interp = RegularGridInterpolator(
                (hrrr_lat, hrrr_lon),
                field_vert[k, :, :],
                bounds_error=False,
                fill_value=np.nan,
            )
            lat_grid, lon_grid = np.meshgrid(grid_lat, grid_lon, indexing="ij")
            pts = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
            result[:, :, k] = interp(pts).reshape(H_out, W_out)

        return result

    def compute_vorticity(self, u: np.ndarray, v: np.ndarray,
                          dx: float, dy: float) -> np.ndarray:
        """从 HRRR 风场用中心差分计算垂直涡度 ζ = ∂v/∂x - ∂u/∂y。

        Args:
            u: 纬向风 (H, W, L)
            v: 经向风 (H, W, L)
            dx: x 方向网格间距 (m)
            dy: y 方向网格间距 (m)

        Returns:
            垂直涡度 ζ (H, W, L)
        """
        H, W, L = u.shape
        zeta = np.zeros_like(u)

        for k in range(L):
            # ∂v/∂x：中心差分
            dv_dx = np.zeros((H, W))
            dv_dx[:, 1:-1] = (v[:, 2:, k] - v[:, :-2, k]) / (2.0 * dx)
            dv_dx[:, 0] = (v[:, 1, k] - v[:, 0, k]) / dx
            dv_dx[:, -1] = (v[:, -1, k] - v[:, -2, k]) / dx

            # ∂u/∂y：中心差分
            du_dy = np.zeros((H, W))
            du_dy[1:-1, :] = (u[2:, :, k] - u[:-2, :, k]) / (2.0 * dy)
            du_dy[0, :] = (u[1, :, k] - u[0, :, k]) / dy
            du_dy[-1, :] = (u[-1, :, k] - u[-2, :, k]) / dy

            zeta[:, :, k] = dv_dx - du_dy

        return zeta

    # ── P2: Environmental diagnostic parameters ──

    def compute_ebwd(self, u_profile: np.ndarray, v_profile: np.ndarray,
                     alt_profile: np.ndarray,
                     eib_agl: float = 0.0, eit_agl: float = 6000.0) -> float:
        """计算有效 bulk 风切变 EBWD (Effective Bulk Wind Difference).

        EBWD = |V(EIT) - V(EIB)|，即有效入流层内地面到顶部的风矢量差大小。
        根据 Thompson et al. (2012)，EBWD ≥ 12.5 m/s 是超级单体阈值。

        Args:
            u_profile: 纬向风垂直廓线 (L,) m/s
            v_profile: 经向风垂直廓线 (L,) m/s
            alt_profile: 高度垂直廓线 (L,) m AGL
            eib_agl: 有效入流底高 (m AGL)，默认地面
            eit_agl: 有效入流顶高 (m AGL)，默认 6km

        Returns:
            EBWD (m/s)
        """
        alt_m = np.asarray(alt_profile, dtype=np.float64)
        u = np.asarray(u_profile, dtype=np.float64)
        v = np.asarray(v_profile, dtype=np.float64)
        u_eib = float(np.interp(eib_agl, alt_m, u))
        v_eib = float(np.interp(eib_agl, alt_m, v))
        u_eit = float(np.interp(eit_agl, alt_m, u))
        v_eit = float(np.interp(eit_agl, alt_m, v))
        return float(np.sqrt((u_eit - u_eib)**2 + (v_eit - v_eib)**2))

    def compute_srh(self, u_profile: np.ndarray, v_profile: np.ndarray,
                    alt_profile: np.ndarray,
                    storm_u: float, storm_v: float,
                    layer_top_agl: float = 500.0) -> float:
        """计算风暴相对螺旋度 SRH (Storm-Relative Helicity).

        SRH = ∫₀^H k̂ · (V - C) × (∂V/∂z) dz
        其中 V=(u,v) 环境风，C=(storm_u,storm_v) 风暴运动矢量。

        Args:
            u_profile: 纬向风垂直廓线 (L,) m/s
            v_profile: 经向风垂直廓线 (L,) m/s
            alt_profile: 高度垂直廓线 (L,) m AGL
            storm_u: 风暴纬向运动速度 m/s
            storm_v: 风暴经向运动速度 m/s
            layer_top_agl: 积分层顶高度 (m AGL)，默认 500m

        Returns:
            SRH (m²/s²)
        """
        alt_m = np.asarray(alt_profile, dtype=np.float64)
        u = np.asarray(u_profile, dtype=np.float64)
        v = np.asarray(v_profile, dtype=np.float64)

        mask = alt_m <= layer_top_agl
        if mask.sum() < 2:
            return 0.0

        alt_layer = alt_m[mask]
        u_layer = u[mask]
        v_layer = v[mask]

        dz = np.diff(alt_layer)
        if len(dz) == 0:
            return 0.0

        # Storm-relative wind at midpoints
        u_mid = (u_layer[:-1] + u_layer[1:]) / 2.0 - storm_u
        v_mid = (v_layer[:-1] + v_layer[1:]) / 2.0 - storm_v

        # ∂v/∂z, ∂u/∂z at midpoints
        dv_dz = np.diff(v_layer) / dz
        du_dz = np.diff(u_layer) / dz

        # k̂ · (V-C) × (∂V/∂z) = u_mid * dv_dz - v_mid * du_dz
        srh = np.sum((u_mid * dv_dz - v_mid * du_dz) * dz)
        return float(srh)

    def compute_all_env_params(self, timestamp: datetime,
                                grid_lon: np.ndarray,
                                grid_lat: np.ndarray,
                                grid_alt: np.ndarray,
                                storm_u: float, storm_v: float,
                                storm_lon: float, storm_lat: float,
                                ) -> dict[str, float]:
        """Compute all environmental diagnostic parameters for a storm.

        Extracts wind profile at storm center from HRRR, then computes
        EBWD and SRH500. More parameters can be added as needed.

        Args:
            timestamp: 观测时间
            grid_lon, grid_lat, grid_alt: GridRad 网格坐标
            storm_u, storm_v: 风暴运动矢量 (来自 GridRad-Severe 追踪)
            storm_lon, storm_lat: 风暴中心位置

        Returns:
            {"ebwd": float, "srh500": float} 等诊断参数字典
        """
        # Extract u,v at storm center (1D vertical profiles)
        fields = self.get_fields(
            timestamp, 
            np.array([storm_lon]), 
            np.array([storm_lat]), 
            grid_alt,
            variables=["u", "v", "w", "t", "q", "p"]
        )

        if "u" not in fields or "v" not in fields:
            return {}

        # Storm center = single grid point → squeeze to 1D profiles
        u_profile = fields["u"].squeeze()  # (L,)
        v_profile = fields["v"].squeeze()  # (L,)

        results = {}

        # EBWD: 0-6km AGL
        try:
            results["ebwd"] = self.compute_ebwd(
                u_profile, v_profile, grid_alt,
                eib_agl=0.0, eit_agl=6000.0
            )
        except Exception:
            results["ebwd"] = 0.0

        # SRH500: 0-500m AGL
        try:
            results["srh500"] = self.compute_srh(
                u_profile, v_profile, grid_alt,
                storm_u=storm_u, storm_v=storm_v,
                layer_top_agl=500.0
            )
        except Exception:
            results["srh500"] = 0.0

        return results
