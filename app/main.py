"""测量网平差 API（FastAPI）。

端点：
* POST /api/v1/adjustments        平差并（可选）落库生成版本
* POST /api/v1/preview            预演：临时剔除观测/调权，与原案对比
* GET  /api/v1/schemes            方案列表
* GET  /api/v1/schemes/{id}/versions                版本列表
* GET  /api/v1/schemes/{id}/versions/{no}          查版本（参数追溯）
* POST /api/v1/schemes/{id}/versions/{no}/recompute 按版本复算
* GET  /api/v1/versions           全部版本
* POST /api/v1/deformations                       多期形变分析并（可选）落库
* GET  /api/v1/deformation-schemes                形变分析方案列表
* GET  /api/v1/deformation-schemes/{id}/versions             版本列表
* GET  /api/v1/deformation-schemes/{id}/versions/{no}       按编号复取
* POST /api/v1/deformation-schemes/{id}/versions/{no}/recompute
* GET  /health, /api/v1/software_versions
"""
from __future__ import annotations

import platform
import sys
from typing import Optional

import numpy as np
import scipy
import fastapi
import pydantic
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import __version__, storage
from .adjustment import AdjustmentError, public_result, run_adjustment
from .deformation import run_deformation
from .models import (
    AdjustmentRequest,
    DeformationRecomputeOptions,
    DeformationRequest,
    PreviewRequest,
)
from .preview import run_preview

app = FastAPI(
    title="测量网平差 API",
    version=__version__,
    description=(
        "外业多站导线/控制点汇总平差：单位统一、网络预检、加权最小二乘、"
        "残差与标准化残差、误差椭圆、粗差判别、预演与版本追溯；"
        "多期形变分析：各期解算、稳定基准对齐、协方差传播、三维位移、"
        "置信椭球、显著性检验、替代基准比较与版本复取。"
    ),
)


@app.exception_handler(AdjustmentError)
async def adjustment_error_handler(request, exc: AdjustmentError):
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "error": exc.code,
            "message": str(exc),
            "details": exc.details,
        },
    )


def software_versions() -> dict[str, str]:
    return {
        "app": __version__,
        "python": sys.version.split()[0],
        "fastapi": fastapi.__version__,
        "pydantic": pydantic.VERSION,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/software_versions")
def get_software_versions():
    return software_versions()


@app.post("/api/v1/adjustments")
def create_adjustment(req: AdjustmentRequest):
    result = run_adjustment(req)
    version_info = None
    if req.save:
        version_info = storage.save_version(req, result, software_versions())
    body = public_result(result)
    body["version"] = version_info
    return body


@app.post("/api/v1/preview")
def preview(req: PreviewRequest):
    return run_preview(req)


@app.get("/api/v1/schemes")
def get_schemes():
    return {"schemes": storage.list_schemes()}


@app.get("/api/v1/versions")
def get_all_versions():
    return {"versions": storage.list_versions()}


@app.get("/api/v1/schemes/{scheme_id}/versions")
def get_versions(scheme_id: int):
    versions = storage.list_versions(scheme_id)
    return {"scheme_id": scheme_id, "versions": versions}


@app.get("/api/v1/schemes/{scheme_id}/versions/{version_no}")
def get_version(scheme_id: int, version_no: int):
    try:
        v = storage.load_version(scheme_id, version_no)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return v


@app.post("/api/v1/schemes/{scheme_id}/versions/{version_no}/recompute")
def recompute_version(scheme_id: int, version_no: int, save: bool = False):
    """按历史版本保存的完整请求复算；默认不产生新版本。"""
    try:
        v = storage.load_version(scheme_id, version_no)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    req = AdjustmentRequest.model_validate(v["request"])
    req.save = save
    result = run_adjustment(req)
    new_version = None
    if save:
        new_version = storage.save_version(req, result, software_versions())
    body = public_result(result)
    body["recomputed_from"] = {
        "scheme_id": scheme_id,
        "version_no": version_no,
        "created_at": v["created_at"],
        "software": v["software"],
        "request_parameters": v["request"],
    }
    body["version"] = new_version
    return body


# ---------------------------------------------------------------------------
# 多期形变分析
# ---------------------------------------------------------------------------

@app.post("/api/v1/deformations")
def create_deformation(req: DeformationRequest):
    """多期形变分析：各期解算、稳定基准对齐、位移/椭球/显著性检验。"""
    result = run_deformation(req)
    version_info = None
    if req.save:
        version_info = storage.save_deformation_version(
            req, result, software_versions())
    result["version"] = version_info
    return result


@app.get("/api/v1/deformation-schemes")
def get_deformation_schemes():
    return {"schemes": storage.list_deformation_schemes()}


@app.get("/api/v1/deformation-schemes/{scheme_id}/versions")
def get_deformation_versions(scheme_id: int):
    versions = storage.list_deformation_versions(scheme_id)
    return {"scheme_id": scheme_id, "versions": versions}


@app.get("/api/v1/deformation-schemes/{scheme_id}/versions/{version_no}")
def get_deformation_version(scheme_id: int, version_no: int):
    try:
        v = storage.load_deformation_version(scheme_id, version_no)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return v


@app.post("/api/v1/deformation-schemes/{scheme_id}/versions/{version_no}/recompute")
def recompute_deformation(
    scheme_id: int,
    version_no: int,
    options: Optional[DeformationRecomputeOptions] = None,
    save: bool = False,
):
    """按版本保存的参数复算形变分析。

    可在请求体中用 ``epoch_selection`` 覆盖期次选择（恰好两期），
    缺省沿用版本保存的参数（连续多期）；``save=true`` 时另存为新版本。
    """
    try:
        v = storage.load_deformation_version(scheme_id, version_no)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    req = DeformationRequest.model_validate(v["request"])
    if options is not None and options.epoch_selection is not None:
        req.epoch_selection = options.epoch_selection
    req.save = save
    result = run_deformation(req)
    new_version = None
    if save:
        new_version = storage.save_deformation_version(
            req, result, software_versions())
    result["recomputed_from"] = {
        "scheme_id": scheme_id,
        "version_no": version_no,
        "created_at": v["created_at"],
        "software": v["software"],
        "request_parameters": v["request"],
    }
    result["version"] = new_version
    return result
