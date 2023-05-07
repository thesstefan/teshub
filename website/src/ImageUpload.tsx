import React, { useState, useEffect } from 'react';

import { ReactCompareSlider, ReactCompareSliderImage, ReactCompareSliderHandle } from 'react-compare-slider';
import './ImageUpload.css';


function ImageUpload() {
    const [loading, setLoading] = useState<boolean>(false)

    const [file, setFile] = useState<File | undefined>(undefined);
    const [inputImageUrl, setInputImageUrl] = useState<string | undefined>(undefined);
    const [segmentationMaskUrl, setSegmentationMaskUrl] = useState<string | undefined>(undefined);

    const handleFileUploaded = (file: File) => {
        const formData = new FormData();
        formData.append('image', file);

        fetch('/preprocess/predict', {
            method: 'POST',
            body: formData,
        }).then(response => response.blob()).then((data) => {
            const segmentationMaskUrl = URL.createObjectURL(data);
            setSegmentationMaskUrl(segmentationMaskUrl);
        }).catch(
            error => console.error(error)
        ).finally(() => {
            setLoading(false);
        });
    }

    useEffect(() => {
        if (file) {
            handleFileUploaded(file)
        }
    }, [file])

    const handleFileUploadClick = (event: React.ChangeEvent<HTMLInputElement>) => {
        setLoading(true);
        const files = event.target.files;

        if (files) {
            const uploadedFile = files[0];
            setFile(uploadedFile);

            const reader = new FileReader();
            reader.onload = (e) => {
                const inputImageUrl = e.target?.result as string;
                setInputImageUrl(inputImageUrl);
            };

            reader.readAsDataURL(uploadedFile);
        }
    };

    return (
        <>
            <div className="image-controls">
                {loading && <div className="loader"></div>}

                {!loading && inputImageUrl && segmentationMaskUrl &&
                    <div className="image-slider-container animate pop">
                        <ReactCompareSlider
                            handle={
                                <ReactCompareSliderHandle
                                    buttonStyle={{
                                        border: 0,
                                        backdropFilter: 'none',
                                        WebkitBackdropFilter: 'none',
                                        boxShadow: 'none',
                                        color: 'black',
                                        height: 0,
                                        gap: 15,
                                    }}
                                    linesStyle={{ width: 8, color: 'black' }} />
                            }

                            itemOne={
                                <ReactCompareSliderImage src={inputImageUrl} />
                            }

                            itemTwo={
                                <ReactCompareSliderImage src={segmentationMaskUrl} />
                            }
                        />
                    </div>
                }
            </div>
            <div className="image-button-box">
                <label htmlFor="image-upload" className="image-upload">
                    Upload Image
                </label>
                <input
                    id="image-upload"
                    type="file"
                    accept="image/jpg, image/jpeg"
                    onChange={handleFileUploadClick}
                />
            </div>
        </>
    )
}

export default ImageUpload;
