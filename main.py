"""
Multi-Model Embedding Service

Sentence-Transformers + BGE-M3 텍스트 임베딩 서비스

주요 기능:
1. /embed: Sentence-Transformers 임베딩
2. /embed_bge_m3: BGE-M3 임베딩
3. /similarity: 유사도 계산
4. /health: 헬스체크
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import os

# BGE-M3용 imports
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# ============================================================================
# 로깅 설정
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# BGE-M3 Embedder 클래스
# ============================================================================

class Embedder:
    """텍스트 임베딩 생성 (Hugging Face Transformers 직접 사용)"""
    
    _model: Optional[AutoModel] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _device: str = None
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """임베딩 모델 로드"""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            print(f"Device: {self._device}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()
            
            print(f"Embedding model loaded")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - 토큰 임베딩의 평균"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    async def embed(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        if not text:
            return []
        
        with torch.no_grad():
            # 토큰화
            encoded_input = self._tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self._device)
            
            # 모델 실행
            model_output = self._model(**encoded_input)
            
            # Mean pooling
            embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # 정규화
            embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy()[0].tolist()
    
    async def embed_batch(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> List[List[float]]:
        """
        배치 텍스트 임베딩
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기
            normalize: 정규화 여부
        
        Returns:
            임베딩 벡터 리스트
        """
        if not texts:
            return []
        
        # 빈 텍스트 필터링
        valid_texts = [t for t in texts if t]
        if not valid_texts:
            return []
        
        print(f"🔢 Embedding {len(valid_texts)} texts...")
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                
                # 토큰화
                encoded_input = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self._device)
                
                # 모델 실행
                model_output = self._model(**encoded_input)
                
                # Mean pooling
                embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
                # 정규화
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
                
                if (i + batch_size) % (batch_size * 10) == 0:
                    print(f"  Progress: {min(i + batch_size, len(valid_texts))}/{len(valid_texts)}")
        
        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        
        print(f"Embeddings generated")
        return all_embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """임베딩 차원 반환"""
        # 더미 텍스트로 차원 확인
        with torch.no_grad():
            encoded_input = self._tokenizer(
                "test",
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self._device)
            
            model_output = self._model(**encoded_input)
            embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            return embedding.shape[1]

# ============================================================================
# FastAPI 앱 초기화
# ============================================================================
app = FastAPI(
    title="Multi-Model Embedding Service",
    description="Sentence-BERT + BGE-M3 텍스트 임베딩 서비스",
    version="1.0.0"
)

# ============================================================================
# 전역 변수: 모델 인스턴스들
# ============================================================================
# 기존 Sentence-Transformers 모델
model: Optional[SentenceTransformer] = None
MODEL_NAME = os.getenv(
    "MODEL_NAME", 
    "sentence-transformers/distiluse-base-multilingual-cased-v2"
)

# 새로 추가: BGE-M3 모델
bge_embedder: Optional[Embedder] = None
BGE_MODEL_NAME = "BAAI/bge-m3"

# ============================================================================
# 요청/응답 스키마 정의
# ============================================================================

class EmbedRequest(BaseModel):
    """임베딩 생성 요청"""
    texts: List[str] = Field(..., description="임베딩할 텍스트 리스트", min_items=1)
    normalize: bool = Field(True, description="임베딩 정규화 여부")

class EmbedResponse(BaseModel):
    """임베딩 생성 응답"""
    embeddings: List[List[float]] = Field(..., description="생성된 임베딩")
    dimension: int = Field(..., description="임베딩 차원")
    count: int = Field(..., description="처리된 텍스트 수")

class SimilarityRequest(BaseModel):
    """유사도 계산 요청"""
    query_text: str = Field(..., description="쿼리 텍스트")
    corpus_embeddings: List[List[float]] = Field(..., description="코퍼스 임베딩")

class SimilarityResponse(BaseModel):
    """유사도 계산 응답"""
    similarities: List[float] = Field(..., description="유사도 점수 리스트")
    count: int = Field(..., description="계산된 유사도 수")

# ============================================================================
# 앱 시작 시 모델 로드
# ============================================================================

@app.on_event("startup")
async def load_model():
    """앱 시작 시 두 개의 임베딩 모델을 로드"""
    global model, bge_embedder
    
    try:
        # 1. 기존 Sentence-Transformers 모델 로드
        logger.info(f"📦 Sentence-Transformers 모델 로딩 시작: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        logger.info(f"✅ Sentence-Transformers 모델 로딩 완료")
        logger.info(f"   차원: {model.get_sentence_embedding_dimension()}")
        
        # 2. BGE-M3 모델 로드
        logger.info(f"📦 BGE-M3 모델 로딩 시작: {BGE_MODEL_NAME}")
        bge_embedder = Embedder(model_name=BGE_MODEL_NAME)
        logger.info(f"✅ BGE-M3 모델 로딩 완료")
        logger.info(f"   차원: {bge_embedder.get_embedding_dimension()}")
        
    except Exception as e:
        logger.error(f"❌ 모델 로딩 실패: {e}")
        raise

# ============================================================================
# API 엔드포인트
# ============================================================================

@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    
    # Sentence-Transformers 상태
    st_status = {
        "status": "healthy" if model is not None else "loading",
        "model_name": MODEL_NAME,
        "embedding_dimension": model.get_sentence_embedding_dimension() if model else None
    }
    
    # BGE-M3 상태
    bge_status = {
        "status": "healthy" if bge_embedder is not None else "loading",
        "model_name": BGE_MODEL_NAME,
        "device": bge_embedder._device if bge_embedder else None,
        "embedding_dimension": bge_embedder.get_embedding_dimension() if bge_embedder else None
    }
    
    return {
        "service": "Multi-Model Embedding Service",
        "models": {
            "sentence_transformers": st_status,
            "bge_m3": bge_status
        }
    }

@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """
    Sentence-Transformers 모델로 임베딩 생성
    
    기존 sentence-transformers 라이브러리 사용
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Sentence-Transformers 모델이 아직 로드되지 않았습니다."
        )
    
    try:
        logger.debug(f"📝 [ST] 임베딩 생성 요청: {len(request.texts)}개 텍스트")
        
        # 임베딩 생성
        embeddings = model.encode(
            request.texts,
            convert_to_numpy=True,
            normalize_embeddings=request.normalize,
            show_progress_bar=False
        )
        
        embeddings_list = embeddings.tolist()
        
        logger.debug(f"✅ [ST] 임베딩 생성 완료: {len(embeddings_list)}개")
        
        return EmbedResponse(
            embeddings=embeddings_list,
            dimension=len(embeddings_list[0]) if embeddings_list else 0,
            count=len(embeddings_list)
        )
        
    except Exception as e:
        logger.error(f"❌ [ST] 임베딩 생성 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"임베딩 생성 중 오류 발생: {str(e)}"
        )

@app.post("/embed_bge_m3", response_model=EmbedResponse)
async def embed_texts_bge_m3(request: EmbedRequest):
    """
    BGE-M3 모델로 임베딩 생성
    
    BAAI/bge-m3 모델 사용 (1024차원)
    Hugging Face Transformers 직접 사용
    """
    if bge_embedder is None:
        raise HTTPException(
            status_code=503,
            detail="BGE-M3 모델이 아직 로드되지 않았습니다."
        )
    
    try:
        logger.debug(f"📝 [BGE-M3] 임베딩 생성 요청: {len(request.texts)}개 텍스트")
        
        # Embedder 클래스의 embed_batch 사용
        embeddings_list = await bge_embedder.embed_batch(
            texts=request.texts,
            batch_size=32,
            normalize=request.normalize
        )
        
        logger.debug(f"✅ [BGE-M3] 임베딩 생성 완료: {len(embeddings_list)}개")
        
        return EmbedResponse(
            embeddings=embeddings_list,
            dimension=len(embeddings_list[0]) if embeddings_list else 0,
            count=len(embeddings_list)
        )
        
    except Exception as e:
        logger.error(f"❌ [BGE-M3] 임베딩 생성 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"임베딩 생성 중 오류 발생: {str(e)}"
        )

@app.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """
    쿼리 텍스트와 코퍼스 임베딩 간의 유사도를 계산
    
    기존 Sentence-Transformers 모델 사용
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="모델이 아직 로드되지 않았습니다."
        )
    
    try:
        logger.debug(f"🔍 유사도 계산 요청: 쿼리 1개 vs 코퍼스 {len(request.corpus_embeddings)}개")
        
        # 쿼리 텍스트를 임베딩으로 변환
        query_embedding = model.encode(
            [request.query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # 코퍼스 임베딩
        corpus_embeddings = np.array(request.corpus_embeddings)
        
        # 코사인 유사도 계산
        similarities = np.dot(corpus_embeddings, query_embedding.T).flatten()
        similarities_list = similarities.tolist()
        
        logger.debug(f"✅ 유사도 계산 완료: {len(similarities_list)}개")
        
        return SimilarityResponse(
            similarities=similarities_list,
            count=len(similarities_list)
        )
        
    except Exception as e:
        logger.error(f"❌ 유사도 계산 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"유사도 계산 중 오류 발생: {str(e)}"
        )

@app.get("/")
async def root():
    """루트 엔드포인트 - 서비스 정보 제공"""
    return {
        "service": "Multi-Model Embedding Service",
        "version": "1.0.0",
        "models": {
            "sentence_transformers": {
                "name": MODEL_NAME,
                "status": "running" if model is not None else "loading"
            },
            "bge_m3": {
                "name": BGE_MODEL_NAME,
                "status": "running" if bge_embedder is not None else "loading"
            }
        },
        "endpoints": {
            "health": "GET /health - 헬스체크",
            "embed": "POST /embed - Sentence-Transformers 임베딩",
            "embed_bge_m3": "POST /embed_bge_m3 - BGE-M3 임베딩",
            "similarity": "POST /similarity - 유사도 계산",
            "docs": "GET /docs - API 문서 (Swagger UI)"
        }
    }


# ============================================================================
# 실행 (개발 환경)
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )