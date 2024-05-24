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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.vectorstore.OracleVectorStore;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(OracleVectorStoreProperties.CONFIG_PREFIX)
public class OracleVectorStoreProperties {

	private static final Logger logger = LoggerFactory.getLogger(OracleVectorStore.class);

	public static final String CONFIG_PREFIX = "spring.ai.vectorstore.oracle";

	private int dimensions = OracleVectorStore.INVALID_EMBEDDING_DIMENSION;

	private OracleVectorStore.OracleIndexType indexType = OracleVectorStore.DEFAULT_INDEX_TYPE;

	private OracleVectorStore.OracleDistanceType distanceType = OracleVectorStore.OracleDistanceType.COSINE;

	private byte accuracy = OracleVectorStore.DEFAULT_ACCURACY;

	private boolean removeExistingVectorStoreTable = false;

	public int getDimensions() {
		return dimensions;
	}

	public void setDimensions(int dimensions) {
		this.dimensions = dimensions;
	}

	public OracleVectorStore.OracleIndexType getIndexType() {
		return indexType;
	}

	public void setIndexType(OracleVectorStore.OracleIndexType createIndexMethod) {
		this.indexType = createIndexMethod;
	}

	public OracleVectorStore.OracleDistanceType getDistanceType() {
		return distanceType;
	}

	public byte getAccuracy() {
		return accuracy;
	}

	public void setDistanceType(OracleVectorStore.OracleDistanceType distanceType) {
		this.distanceType = distanceType;
	}

	public boolean isRemoveExistingVectorStoreTable() {
		return removeExistingVectorStoreTable;
	}

	public void setRemoveExistingVectorStoreTable(boolean removeExistingVectorStoreTable) {
		this.removeExistingVectorStoreTable = removeExistingVectorStoreTable;
	}

	public void setAccuracy(byte accuracy) {
		if (accuracy < 0 || accuracy > 100) {
			logger.warn("Invalid accuracy value provided, falling back to default value.");
			this.accuracy = OracleVectorStore.DEFAULT_ACCURACY;
		}
		else {
			this.accuracy = accuracy;
		}
	}

}
