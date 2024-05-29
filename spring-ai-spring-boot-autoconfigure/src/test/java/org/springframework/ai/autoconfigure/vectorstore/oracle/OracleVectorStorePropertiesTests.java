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
package org.springframework.ai.autoconfigure.vectorstore.oracle;

import org.junit.jupiter.api.Test;
import org.springframework.ai.autoconfigure.vectorstore.pgvector.PgVectorStoreProperties;
import org.springframework.ai.vectorstore.OracleVectorStore;
import org.springframework.ai.vectorstore.OracleVectorStore.OracleDistanceType;
import org.springframework.ai.vectorstore.OracleVectorStore.OracleIndexType;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * @author Christian Tzolov
 * @author Fernanda Meheust
 */
public class OracleVectorStorePropertiesTests {

	@Test
	public void defaultValues() {
		var props = new OracleVectorStoreProperties();
		assertThat(props.getDimensions()).isEqualTo(OracleVectorStore.INVALID_EMBEDDING_DIMENSION);
		assertThat(props.getDistanceType()).isEqualTo(OracleDistanceType.COSINE);
		assertThat(props.getIndexType()).isEqualTo(OracleIndexType.NONE);
		assertThat(props.isRemoveExistingVectorStoreTable()).isFalse();
		assertThat(props.getAccuracy()).isEqualTo(OracleVectorStore.DEFAULT_ACCURACY);
	}

	@Test
	public void customValues() {
		var props = new OracleVectorStoreProperties();

		props.setDimensions(1536);
		props.setDistanceType(OracleDistanceType.EUCLIDEAN);
		props.setIndexType(OracleIndexType.HNSW);
		props.setRemoveExistingVectorStoreTable(true);
		props.setAccuracy((byte) 100);

		assertThat(props.getAccuracy()).isEqualTo((byte) 100);
		assertThat(props.getDimensions()).isEqualTo(1536);
		assertThat(props.getDistanceType()).isEqualTo(OracleDistanceType.EUCLIDEAN);
		assertThat(props.getIndexType()).isEqualTo(OracleIndexType.HNSW);
		assertThat(props.isRemoveExistingVectorStoreTable()).isTrue();

		props.setAccuracy((byte) 120);
		assertThat(props.getAccuracy()).isEqualTo(OracleVectorStore.DEFAULT_ACCURACY);
	}

}
