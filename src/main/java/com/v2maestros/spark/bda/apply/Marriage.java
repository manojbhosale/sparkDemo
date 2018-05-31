/** File provided by V2 Maestros for its students for learning purposes only
 * Copyright @2016 All rights reserved.
 */
package com.v2maestros.spark.bda.apply;

public class Marriage {

	/**
	 * @return the marriageId
	 */
	public Double getMarriageId() {
		return marriageId;
	}

	/**
	 * @param marriageId the marriageId to set
	 */
	public void setMarriageId(Double marriageId) {
		this.marriageId = marriageId;
	}

	/**
	 * @return the marriageName
	 */
	public String getMarriageName() {
		return marriageName;
	}

	/**
	 * @param marriageName the marriageName to set
	 */
	public void setMarriageName(String marriageName) {
		this.marriageName = marriageName;
	}

	Double marriageId;
	String marriageName;
	
	public Marriage(Double marriageId, String marriageName) {
		this.marriageId=marriageId;
		this.marriageName=marriageName;
	}
}
