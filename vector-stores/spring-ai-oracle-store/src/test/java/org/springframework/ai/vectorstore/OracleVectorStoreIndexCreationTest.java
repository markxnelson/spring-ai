/*
 * Copyright 2023 - 2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.springframework.ai.vectorstore;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.Mockito;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.jdbc.core.JdbcTemplate;

/**
 * @author Fernanda Meheust
 */
public class OracleVectorStoreIndexCreationTest {

	@ParameterizedTest
	@CsvSource({ "COSINE,HNSW,70", "DOT,HNSW,70", "EUCLIDEAN,HNSW,70", "COSINE,IVF,70", "DOT,IVF,70",
			"EUCLIDEAN,IVF,70", "COSINE,HNSW,110", "DOT,HNSW,110", "EUCLIDEAN,HNSW,110", "COSINE,IVF,-1", "DOT,IVF,-1",
			"EUCLIDEAN,IVF,-1" })
	public void testDotConfig(String sDistanceType, String sIndexType, byte accuracy) throws Exception {
		JdbcTemplate jdbcTemplate = Mockito.mock(JdbcTemplate.class);
		EmbeddingModel embeddingModel = Mockito.mock(EmbeddingModel.class);

		OracleVectorStore.OracleDistanceType distanceType = OracleVectorStore.OracleDistanceType.valueOf(sDistanceType);
		OracleVectorStore.OracleIndexType indexType = OracleVectorStore.OracleIndexType.valueOf(sIndexType);
		String indexTypeSQL = switch (indexType) {
			case HNSW -> "ORGANIZATION INMEMORY NEIGHBOR GRAPH";
			case IVF -> "ORGANIZATION NEIGHBOR PARTITIONS";
			default -> "";
		};

		OracleVectorStore vectorStore = new OracleVectorStore(jdbcTemplate, embeddingModel,
				OracleVectorStore.INVALID_EMBEDDING_DIMENSION, distanceType, false, indexType, accuracy);
		vectorStore.afterPropertiesSet();
		Mockito.verify(jdbcTemplate).execute(Mockito.contains(indexTypeSQL));
		Mockito.verify(jdbcTemplate).execute(Mockito.contains(distanceType.toString()));
		if (accuracy < 0 || accuracy > 100) {
			Mockito.verify(jdbcTemplate).execute(Mockito.contains(String.valueOf(OracleVectorStore.DEFAULT_ACCURACY)));
		}
		else {
			Mockito.verify(jdbcTemplate).execute(Mockito.contains(String.valueOf(accuracy)));
		}
	}

}
