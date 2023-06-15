import React, { useState, useEffect } from 'react';

import RangeInput from '../common/RangeInput';
import LegendItem from '../common/LegendItem';

import { ReactCompareSlider, ReactCompareSliderImage, ReactCompareSliderHandle } from 'react-compare-slider';
import './UploadPage.css';

interface WeatherLabels {
    readonly snowy: number;
    readonly foggy: number;
    readonly rainy: number;
    readonly cloudy: number;
}

interface Name2NameMapping {
    [key: string]: string
}

interface ColorPallete {
    readonly [key: string]: number[];
}

const FORMATTED_WEATHER_CUES: Name2NameMapping = {
    background: "Background",
    black_clouds: "Black Clouds",
    white_clouds: "White Clouds",
    blue_sky: "Blue Sky",
    gray_sky: "Gray Sky",
    white_sky: "White Sky",
    fog: "Fog",
    sun: "Sun",
    snow: "Snow",
    shadow: "Shadow",
    wet_ground: "Wet Ground",
    shadow_snow: "Shadowed Snow",
};

function UploadPage() {
    const [loading, setLoading] = useState<boolean>(false)
    const [translationLoading, setTranslationLoading] = useState<boolean>(false)

    const [colorMap, setColorPallete] = useState<ColorPallete | undefined>(undefined);
    const [labels, setLabels] = useState<WeatherLabels | undefined>(undefined);
    const [file, setFile] = useState<File | undefined>(undefined);
    const [segmentation, setSegmentationFile] = useState<File | undefined>(undefined);
    const [inputImageUrl, setInputImageUrl] = useState<string | undefined>(undefined);
    const [segmentationMaskUrl, setSegmentationMaskUrl] = useState<string | undefined>(undefined);
    const [translatedUrl, setTranslatedUrl] = useState<string | undefined>(undefined);
    const [viewUrl, setViewUrl] = useState<string | undefined>(undefined);

    useEffect(() => {
        if (labels) {
            console.log(labels)
        }
        if (colorMap) {
            console.log(colorMap)
        }
    }, [labels, colorMap])

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

    const handleFileUploaded = (file: File) => {
        const formData = new FormData();
        formData.append('image', file);

        fetch('/predict/segmentation', {
            method: 'POST',
            body: formData,
        }).then(response => {
            const labelsJson = response.headers.get('labels')!;
            const colorPalleteJson = response.headers.get('color_map')!;

            setLabels(JSON.parse(labelsJson))
            setColorPallete(JSON.parse(colorPalleteJson))

            console.log(JSON.parse(colorPalleteJson))

            return response.blob();
        }).then((data) => {
            const segmentationMaskUrl = URL.createObjectURL(data);
            setSegmentationMaskUrl(segmentationMaskUrl);
            setViewUrl(segmentationMaskUrl);

            const file = new File([data], "segmentation.png", { type: 'image/png' });
            setSegmentationFile(file);
        }).catch(
            error => console.error(error)
        ).finally(() => {
            setLoading(false);
        });
    }

    const handleTranslationClick = (
        type: string,
        event: React.MouseEvent<HTMLButtonElement>
    ) => {
        setTranslationLoading(true);

        const formData = new FormData();
        formData.append('image', file!);
        formData.append('seg', segmentation!);

        fetch(`/predict/translation/${type}`, {
            method: 'POST',
            body: formData,
        }).then((response) => response.blob()
        ).then((data) => {
            const translatedUrl = URL.createObjectURL(data);
            setTranslatedUrl(translatedUrl);

            setViewUrl(translatedUrl);
        }).catch(
            error => console.error(error)
        ).finally(() => {
            setTranslationLoading(false);
        });
    }

    const handleAddSnowClick = (event: React.MouseEvent<HTMLButtonElement>) => {
        handleTranslationClick('add_snow', event);
    }
    const handleAddFogClick = (event: React.MouseEvent<HTMLButtonElement>) => {
        handleTranslationClick('add_fog', event);
    }
    const handleAddCloudsClick = (event: React.MouseEvent<HTMLButtonElement>) => {
        handleTranslationClick('add_clouds', event);
    }
    const handleShowSegmentationClick = (event: React.MouseEvent<HTMLButtonElement>) => {
        setViewUrl(segmentationMaskUrl)
    }

    return (
        <>
            <div className="image-button-box">
                {inputImageUrl && segmentationMaskUrl && translationLoading &&
                    <div className="translation-loader-cont">
                        <div className="translation-loader"></div>
                    </div>

                }
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
            <div className="image-controls">
                {loading && <div className="loader"></div>}

                {!loading &&
                    <div className="show">
                        <div className="prediction-results">
                            <div className="label-results">
                                {labels &&
                                    <div className="weather-label-container animate pop">
                                        <div className="weather-label">
                                            <div className="weather-label-name">Cloudy</div>
                                            <div className="weather-label-value">{labels!.cloudy.toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 })}</div>
                                        </div>
                                        <div className="weather-label">
                                            <div className="weather-label-name">Rainy</div>
                                            <div className="weather-label-value">{labels!.rainy.toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 })}</div>
                                        </div>
                                        <div className="weather-label">
                                            <div className="weather-label-name">Foggy</div>
                                            <div className="weather-label-value">{labels!.foggy.toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 })}</div>
                                        </div>
                                        <div className="weather-label">
                                            <div className="weather-label-name">Snowy</div>
                                            <div className="weather-label-value">{labels!.snowy.toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 })}</div>
                                        </div>
                                    </div>
                                }
                            </div>
                            <div className="image-results">
                                {inputImageUrl && viewUrl &&
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
                                                <ReactCompareSliderImage src={viewUrl} />
                                            }
                                        />
                                    </div>
                                }
                            </div>
                            {colorMap &&
                                <div>
                                    <div className="weather-cue-legend">
                                        {Object.entries(colorMap).map(([name, color]) => {
                                            return <LegendItem name={FORMATTED_WEATHER_CUES[name]} color={`rgb(${color.join(',')})`} />
                                        })}
                                    </div>
                                </div>
                            }
                        </div>
                        {inputImageUrl && segmentationMaskUrl && colorMap &&
                            <div className="translation-buttons">
                                <button
                                    className="button-50 translation-button"
                                    id="add-snow-button"
                                    onClick={handleAddSnowClick}
                                >
                                    Add Snow
                                </button>
                                <button
                                    className="translation-button button-50"
                                    onClick={handleAddCloudsClick}
                                >
                                    Add Clouds
                                </button>
                                <button
                                    className="translation-button button-50"
                                    onClick={handleAddFogClick}
                                >
                                    Add Fog
                                </button>
                                <button
                                    className="translation-button button-50"
                                    onClick={handleShowSegmentationClick}
                                >
                                    Show Segmentation
                                </button>
                            </div>
                        }
                    </div>
                }
            </div>
        </>
    )
}

export default UploadPage;
