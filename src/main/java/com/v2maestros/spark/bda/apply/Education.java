/** File provided by V2 Maestros for its students for learning purposes only
 * Copyright @2016 All rights reserved.
 */
package com.v2maestros.spark.bda.apply;

public class Education {

	/**
	 * @return the eduId
	 */
	public Double getEduId() {
		return eduId;
	}

	/**
	 * @param eduId the eduId to set
	 */
	public void setEduId(Double eduId) {
		this.eduId = eduId;
	}

	/**
	 * @return the eduName
	 */
	public String getEduName() {
		return eduName;
	}

	/**
	 * @param eduName the eduName to set
	 */
	public void setEduName(String eduName) {
		this.eduName = eduName;
	}

	Double eduId;
	String eduName;
	
	public Education(Double eduId, String eduName) {
		this.eduId=eduId;
		this.eduName=eduName;
	}
}
